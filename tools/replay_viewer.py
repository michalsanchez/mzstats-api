import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

PITCH_WIDTH = 750
PITCH_HEIGHT = 1000


def draw_pitch(ax):

    ax.add_patch(Rectangle((0, 0), PITCH_WIDTH, PITCH_HEIGHT, fill=False, linewidth=1))

    ax.plot([0, PITCH_WIDTH], [PITCH_HEIGHT/2, PITCH_HEIGHT/2], linewidth=0.8)

    ax.add_patch(Circle((PITCH_WIDTH/2, PITCH_HEIGHT/2), 60, fill=False, linewidth=0.8))

    ax.add_patch(Rectangle((PITCH_WIDTH/2-200, 0), 400, 120, fill=False, linewidth=0.8))
    ax.add_patch(Rectangle((PITCH_WIDTH/2-200, PITCH_HEIGHT-120), 400, 120, fill=False, linewidth=0.8))

    ax.set_xlim(0, PITCH_WIDTH)
    ax.set_ylim(0, PITCH_HEIGHT)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")


def load_frames(match_id):

    path = f"data_d_and_p/match_{match_id}.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "frames" in data:
        return data["frames"]

    return data


def main():

    if len(sys.argv) < 2:
        print("Použitie:")
        print("python tools/mz_replay_viewer.py MATCH_ID")
        return

    match_id = sys.argv[1]

    frames = load_frames(match_id)

    print("Počet framov:", len(frames))

    fig, ax = plt.subplots(figsize=(7, 12))

    draw_pitch(ax)

    teamA_artist = ax.scatter([], [], s=120, edgecolors="k", linewidth=0.6, zorder=3)
    teamB_artist = ax.scatter([], [], s=120, edgecolors="k", linewidth=0.6, zorder=3)

    ball_artist = ax.scatter([], [], s=80, color="black", zorder=4)

    ball_text = ax.text(10, 30, "", fontsize=11)

    annotations = {}

    team_ids = set()

    for p in frames[0]["players"]:
        team_ids.add(p["teamId"])

    team_ids = list(team_ids)

    teamA = team_ids[0]
    teamB = team_ids[1] if len(team_ids) > 1 else None

    def safe_set_offsets(scatter, xs, ys):

        if len(xs) == 0:
            scatter.set_offsets(np.empty((0,2)))
        else:
            scatter.set_offsets(np.column_stack((xs, ys)))

    def update(frame_index):

        frame = frames[frame_index]

        A_x = []
        A_y = []

        B_x = []
        B_y = []

        for p in frame["players"]:

            iid = p["internalId"]
            x = p["x"]
            y = p["y"]
            team = p["teamId"]

            if team == teamA:
                A_x.append(x)
                A_y.append(y)
            else:
                B_x.append(x)
                B_y.append(y)

            label = str(p.get("shirt", iid))

            if iid not in annotations:
                annotations[iid] = ax.text(
                    x, y, label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    zorder=5
                )
            else:
                annotations[iid].set_position((x, y))

        safe_set_offsets(teamA_artist, A_x, A_y)
        safe_set_offsets(teamB_artist, B_x, B_y)

        teamA_artist.set_facecolor("red")
        teamB_artist.set_facecolor("blue")

        bx = frame["ball"]["x"]
        by = frame["ball"]["y"]

        ball_artist.set_offsets([[bx, by]])

        clock = frame.get("clock", 0)

        minute = int(clock / 60)
        second = int(clock % 60)

        ball_text.set_text(
            f"Frame: {frame_index}\n"
            f"Time: {minute}:{second:02d}\n"
            f"Ball X: {bx:.2f}\n"
            f"Ball Y: {by:.2f}"
        )

        ax.set_title(f"Match replay viewer")

        return teamA_artist, teamB_artist, ball_artist, ball_text, *annotations.values()

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=30,
        blit=False
    )

    plt.show()


if __name__ == "__main__":
    main()
