import json
import sys
from statistics import median


def detect_formation(players):

    if not players:
        return "unknown"

    ys = sorted(p["y"] for p in players)

    threshold = 80
    lines = []
    current = [ys[0]]

    for y in ys[1:]:

        if abs(y - median(current)) < threshold:
            current.append(y)

        else:
            lines.append(len(current))
            current = [y]

    lines.append(len(current))

    return "-".join(str(x) for x in lines)


def main(match_id, frame_index):

    path = f"data_d_and_p/match_{match_id}.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]
    teams = data["teams"]

    frame = frames[int(frame_index)]

    print("\nFRAME:", frame["frame"], "CLOCK:", frame["clock"])
    print("-----------------------------------")

    for team_id_str, team in teams.items():

        team_id = int(team_id_str)

        players = [
            p for p in frame["players"]
            if p["teamId"] == team_id and not p.get("goalkeeper")
        ]

        formation = detect_formation(players)

        print("\nTEAM:", team["name"])
        print("FORMATION:", formation)
        print("PLAYERS:", len(players))

        for p in players:
            print(
                f'{p["name"]}   x={p["x"]}   y={p["y"]}'
            )


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python tools/inspect_frame.py <match_id> <frame>")
        sys.exit()

    match_id = sys.argv[1]
    frame = int(sys.argv[2])

    main(match_id, frame)
