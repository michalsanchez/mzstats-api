#!/usr/bin/env python3

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

PITCH_HEIGHT = 1000
PITCH_WIDTH = 750

KICK_POINTS = [
    (292, 55),
    (453, 55),
    (292, 946),
    (453, 946)
]

TOLERANCE = 5


def near(a, b):
    return abs(a - b) <= TOLERANCE


def load_match(match_id):

    path = f"data_d_and_p/match_{match_id}.json"

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_goal_kick(ball):

    for x, y in KICK_POINTS:
        if near(ball["x"], x) and near(ball["y"], y):
            return True

    return False


# -----------------------------
# REAL MATCH TIME
# -----------------------------

def frame_to_match_time(clock):

    seconds = int(clock)

    m = seconds // 60
    s = seconds % 60

    return f"{m:02d}:{s:02d}"


# -----------------------------
# FREEZE FRAME DETECTION
# -----------------------------

def find_goal_kick_frames(frames):

    result = []

    prev_ball = None
    kick_active = False
    last_static_frame = None

    for f in frames:

        ball = f["ball"]
        on_spot = is_goal_kick(ball)

        if on_spot and not kick_active:

            kick_active = True
            prev_ball = ball
            last_static_frame = f
            continue

        if kick_active:

            if ball["x"] == prev_ball["x"] and ball["y"] == prev_ball["y"]:
                last_static_frame = f
            else:
                result.append(last_static_frame)
                kick_active = False

        prev_ball = ball

    return result


def find_goalkeeper(players):

    for p in players:
        if p["goalkeeper"]:
            return p

    return None


# -----------------------------
# NORMALIZE ATTACK DIRECTION
# -----------------------------

def normalize_direction(players, team_id, frame_number, total_frames, attack_dir):

    second_half = frame_number > total_frames / 2

    start_gk_y = attack_dir[team_id]

    flip = False

    if start_gk_y < PITCH_HEIGHT / 2:
        flip = True

    if second_half:
        flip = not flip

    if flip:
        for p in players:
            p["y"] = PITCH_HEIGHT - p["y"]

    return players


# -----------------------------
# FORMATION DETECTION
# -----------------------------

def detect_formation(players):

    ys = np.array([[p["y"]] for p in players])

    model = KMeans(n_clusters=3, n_init=20)
    labels = model.fit_predict(ys)

    clusters = {}

    for i, p in enumerate(players):
        clusters.setdefault(labels[i], []).append(p)

    lines = list(clusters.values())

    lines.sort(key=lambda line: np.mean([p["y"] for p in line]))

    attack = lines[0]
    midfield = lines[1]
    defense = lines[2]

    return len(defense), len(midfield), len(attack)


# -----------------------------
# DRAW FRAME
# -----------------------------

def draw_frame(frame):

    os.makedirs("reports/frames", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 8))

    ax.set_xlim(0, PITCH_WIDTH)
    ax.set_ylim(0, PITCH_HEIGHT)

    ax.set_title(f"Frame {frame['frame']}")

    team_colors = {}

    for p in frame["players"]:

        team = str(p["teamId"])

        if team not in team_colors:
            team_colors[team] = "blue" if len(team_colors) == 0 else "red"

        color = team_colors[team]

        if p["goalkeeper"]:
            ax.scatter(p["x"], p["y"], c=color, s=120, marker="s")
        else:
            ax.scatter(p["x"], p["y"], c=color, s=60)

        ax.text(p["x"], p["y"], str(p["shirt"]), fontsize=8)

    ball = frame["ball"]

    ax.scatter(ball["x"], ball["y"], c="yellow", s=80, edgecolors="black")

    ax.invert_yaxis()

    plt.savefig(f"reports/frames/frame_{frame['frame']}.png")
    plt.close()


# -----------------------------
# ANALYZE TEAM
# -----------------------------

def analyze_team(players, team_id, frame_number, total_frames, attack_dir):

    gk = find_goalkeeper(players)

    players = normalize_direction(
        players,
        team_id,
        frame_number,
        total_frames,
        attack_dir
    )

    field_players = [p for p in players if not p["goalkeeper"]]

    defenders, midfielders, attackers = detect_formation(field_players)

    formation = f"{defenders}-{midfielders}-{attackers}"

    return formation


# -----------------------------
# MAIN
# -----------------------------

def main(match_id):

    data = load_match(match_id)

    frames = data["frames"]
    total_frames = data["totalFrames"]

    goal_kick_frames = find_goal_kick_frames(frames)

    attack_dir = {}

    first_frame = goal_kick_frames[0]

    teams = {}

    for p in first_frame["players"]:
        teams.setdefault(str(p["teamId"]), []).append(p)

    for team_id, players in teams.items():

        gk = find_goalkeeper(players)

        attack_dir[team_id] = gk["y"]

    last_confirmed = {}
    candidate = {}

    for f in goal_kick_frames:

        draw_frame(f)

        teams = {}

        for p in f["players"]:
            teams.setdefault(str(p["teamId"]), []).append(p)

        current = {}

        for team_id, players in teams.items():

            formation = analyze_team(
                players,
                team_id,
                f["frame"],
                total_frames,
                attack_dir
            )

            current[team_id] = formation

        time_str = frame_to_match_time(f["clock"])

        if not last_confirmed:

            print(f"\nSTART {time_str}")

            for t in current:
                print(f"{t} -> {current[t]}")

            last_confirmed = current
            continue

        for team in current:

            if current[team] != last_confirmed.get(team):

                if candidate.get(team) == current[team]:

                    print(f"\nTACTIC CHANGE {time_str}")
                    print(f"{team} -> {current[team]}")

                    last_confirmed[team] = current[team]
                    candidate[team] = None

                else:
                    candidate[team] = current[team]


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("python tools/analyze_formations.py MATCH_ID")
        sys.exit()

    main(sys.argv[1])   
