#!/usr/bin/env python3

import json
import sys
import os

PITCH_HEIGHT = 1000

KICK_POINTS = [
    (292, 55),
    (453, 55),
    (292, 946),
    (453, 946)
]

TOLERANCE = 5
MAX_TEST_FRAMES = 4


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


def find_goal_kick_frames(frames):

    result = []
    prev_ball = None
    kick_active = False

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

                if len(result) >= MAX_TEST_FRAMES:
                    break

        prev_ball = ball

    return result


def find_goalkeeper(players):

    for p in players:
        if p["goalkeeper"]:
            return p

    return None


def normalize_direction(players, gk):

    if gk["y"] < PITCH_HEIGHT / 2:
        for p in players:
            p["y"] = PITCH_HEIGHT - p["y"]

    return players


def debug_team(players):

    gk = find_goalkeeper(players)

    players = normalize_direction(players, gk)

    print(f"GK shirt={gk['shirt']} x={gk['x']} y={gk['y']}")

    field = [p for p in players if not p["goalkeeper"]]

    field.sort(key=lambda p: p["y"])

    print("FIELD PLAYERS (sorted by y):")

    for p in field:
        print(
            f"shirt={p['shirt']:>2} "
            f"x={p['x']:>4} "
            f"y={p['y']:>4}"
        )


def analyze_frame(frame):

    teams = {}

    for p in frame["players"]:
        teams.setdefault(str(p["teamId"]), []).append(p)

    for team_id, players in teams.items():

        print(f"\nTEAM {team_id}")

        debug_team(players)


def main(match_id):

    data = load_match(match_id)

    frames = data["frames"]

    goal_kick_frames = find_goal_kick_frames(frames)

    for f in goal_kick_frames:

        print("\n===================================")
        print(f"Frame {f['frame']} | clock {f['clock']}")
        print("===================================")

        analyze_frame(f)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("python tools/debug_goal_kick_positions.py MATCH_ID")
        sys.exit()

    main(sys.argv[1])
