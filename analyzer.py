import os
import copy
import math
import json
import textwrap
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

PITCH_WIDTH = 750
PITCH_HEIGHT = 1000
PENALTY_DEPTH = int(PITCH_HEIGHT * 0.12)
CENTER_RADIUS = int(PITCH_HEIGHT * 0.12)
HALF_TIME = 45 * 60
EUCLIDEAN_MATCH_THRESHOLD = 140.0
LINE_STABILITY_THRESHOLD = 0.85
DEBOUNCE_REQUIRED = 3
MIN_CHANGE_GAP = 300
BASE_COLORS = ["#d7263d", "#0b66c3"]

KICK_POINTS = [
    (292, 55), (453, 55), (292, 946), (453, 946)
]

TOL = 5

def update(status_callback, step, message):
    if status_callback:
        status_callback(step, message)

def near(a, b):
    return abs(a - b) <= TOL

def load_match(match_id):
    path = f"data_d_and_p/match_{match_id}.json"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Match file not found: {path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def frame_time(clock):
    s = int(clock)
    return f"{s//60:02d}:{s%60:02d}"

def is_goal_kick(ball):
    for x, y in KICK_POINTS:
        if near(ball["x"], x) and near(ball["y"], y):
            return True
    return False

def find_freeze_frames(frames):
    res = []
    prev = None
    active = False
    last_static = None

    for f in frames:
        b = f["ball"]
        on_spot = is_goal_kick(b)

        if on_spot and not active:
            active = True
            prev = b
            last_static = f
            continue

        if active:
            if b["x"] == prev["x"] and b["y"] == prev["y"]:
                last_static = f
            else:
                res.append(last_static)
                active = False

        prev = b

    return res

def find_goalkeeper(players):
    for p in players:
        if p.get("goalkeeper"):
            return p
    return None

def map_team_visual_band(players, gk_bottom):
    pts = [dict(p) for p in players]
    field = [p for p in pts if not p.get("goalkeeper")]

    if not field:
        for p in pts:
            p["display_x"] = p["x"]
            p["display_y"] = p["y"]
        return pts

    ys = [p["y"] for p in field]
    miny = min(ys)
    maxy = max(ys)

    if maxy - miny < 1:
        maxy = miny + 1

    top_min = PENALTY_DEPTH + 8
    top_max = PITCH_HEIGHT / 2 - 8
    bot_min = PITCH_HEIGHT / 2 + 8
    bot_max = PITCH_HEIGHT - PENALTY_DEPTH - 8

    for p in pts:
        p["display_x"] = p["x"]

        if p.get("goalkeeper"):
            if gk_bottom:
                p["display_y"] = PITCH_HEIGHT - PENALTY_DEPTH / 2
            else:
                p["display_y"] = PENALTY_DEPTH / 2
        else:
            out_min, out_max = (bot_min, bot_max) if gk_bottom else (top_min, top_max)
            p["display_y"] = out_min + (p["y"] - miny) / (maxy - miny) * (out_max - out_min)

    return pts

def detect_formation_assignment(field_players, gk_bottom):
    if not field_players:
        return "0-0-0", {}

    ys = np.array([[p["display_y"]] for p in field_players])
    model = KMeans(n_clusters=3, n_init=20, random_state=0)
    labels = model.fit_predict(ys)

    clusters = {}

    for i, p in enumerate(field_players):
        clusters.setdefault(labels[i], []).append(p)

    centroids = {}

    for k, v in clusters.items():
        centroids[k] = np.mean([x["display_y"] for x in v])

    ordered = sorted(clusters.items(), key=lambda x: centroids[x[0]])
    groups = [g for _, g in ordered]

    if gk_bottom:
        attack, mid, defense = groups
    else:
        defense, mid, attack = groups

    formation = f"{len(defense)}-{len(mid)}-{len(attack)}"
    assign = {}

    for p in defense:
        assign[p["shirt"]] = "def"
    for p in mid:
        assign[p["shirt"]] = "mid"
    for p in attack:
        assign[p["shirt"]] = "att"

    return formation, assign

def hungarian_assign(prev_players, cur_players):
    if not prev_players or not cur_players:
        return {}, {}

    n = max(len(prev_players), len(cur_players))
    cost = np.full((n, n), 1e6)

    for i, p in enumerate(prev_players):
        for j, q in enumerate(cur_players):
            cost[i, j] = math.hypot(p["x"] - q["x"], p["y"] - q["y"])

    r, c = linear_sum_assignment(cost)
    mapping = {}
    dists = {}

    for i, j in zip(r, c):
        if i < len(prev_players) and j < len(cur_players):
            mapping[prev_players[i]["shirt"]] = cur_players[j]["shirt"]
            dists[prev_players[i]["shirt"]] = cost[i, j]

    return mapping, dists

def detect_wing(attackers_raw, gk_bottom):
    if not attackers_raw:
        return "útok stredom"

    xs = [p["x"] for p in attackers_raw]
    left = sum(x < PITCH_WIDTH * 0.25 for x in xs)
    right = sum(x > PITCH_WIDTH * 0.75 for x in xs)

    if not gk_bottom:
        left, right = right, left

    if left >= 2 and right >= 2:
        return "obe krídla"
    if left >= 2:
        return "ľavé krídlo"
    if right >= 2:
        return "pravé krídlo"

    return "útok stredom"

def draw_pitch(ax):
    ax.add_patch(
        patches.Rectangle((0, 0), PITCH_WIDTH, PITCH_HEIGHT, facecolor="#4caf50")
    )

    stripe = PITCH_HEIGHT / 10

    for i in range(10):
        if i % 2 == 0:
            ax.add_patch(
                patches.Rectangle((0, i * stripe), PITCH_WIDTH, stripe, facecolor="#43a047")
            )

    ax.plot([0, PITCH_WIDTH], [PITCH_HEIGHT / 2, PITCH_HEIGHT / 2], color="white", linewidth=2)

    ax.add_patch(
        patches.Circle((PITCH_WIDTH / 2, PITCH_HEIGHT / 2), CENTER_RADIUS, fill=False, edgecolor="white", linewidth=2)
    )

def draw_change_box(fig, text):
    if not text:
        return

    wrapped = "\n".join(textwrap.wrap(text, width=70))

    fig.text(
        0.5,
        0.94,
        wrapped,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#b00000",
        bbox=dict(facecolor="#ffeaea", edgecolor="#cc0000", boxstyle="round,pad=0.6")
    )

def draw_snapshot(frame, teams, teams_meta, color_map, title, change_text=None):
    os.makedirs("reports/tactics", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 11))
    plt.subplots_adjust(top=0.90)

    ax.set_xlim(0, PITCH_WIDTH)
    ax.set_ylim(0, PITCH_HEIGHT)
    ax.axis("off")

    fig.suptitle(title, fontsize=16, y=0.985)

    draw_pitch(ax)

    if change_text:
        draw_change_box(fig, change_text)

    for tid, players in teams.items():
        meta = teams_meta[tid]
        color = color_map[tid]

        gk = find_goalkeeper(players)
        label_y = PITCH_HEIGHT * 0.95 if gk and gk["y"] > PITCH_HEIGHT / 2 else PITCH_HEIGHT * 0.05
        label = f"{meta['name']} | {meta['formation']} | {meta['wing']}"

        ax.text(
            PITCH_WIDTH / 2,
            label_y,
            label,
            ha="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.18, boxstyle="round,pad=0.5")
        )

        for p in players:
            if p.get("goalkeeper"):
                continue

            ax.scatter(p["display_x"], p["display_y"], c=color, s=160, edgecolors="black")

            ax.text(
                p["display_x"],
                p["display_y"] - 40,
                str(p["shirt"]),
                ha="center",
                fontsize=28,
                fontweight="bold"
            )

    out = f"reports/tactics/frame_{frame['frame']}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()

    return "/" + out

def run_analysis(match_id, team_id=None, full_pitch=False, status_callback=None):
    update(status_callback, "load_match", "📂 Načítavam zápasové dáta...")
    data = load_match(match_id)
    frames = data["frames"]
    teams_info = data["teams"]

    update(status_callback, "freeze_frames", "🎯 Vyhľadávam kľúčové herné situácie...")
    freeze_frames = find_freeze_frames(frames)

    if not freeze_frames:
        update(status_callback, "done", "⚠️ Nenašli sa žiadne taktické situácie")
        return []

    first = freeze_frames[0]
    team_order = []

    for p in first["players"]:
        tid = str(p["teamId"])
        if tid not in team_order:
            team_order.append(tid)

    color_map = {tid: BASE_COLORS[i % len(BASE_COLORS)] for i, tid in enumerate(team_order)}
    images = []
    prev_form = {}
    prev_assign = {}
    prev_positions = {}
    last_change_clock = 0
    total = len(freeze_frames)

    for i, f in enumerate(freeze_frames):
        update(status_callback, "analyzing", f"🧠 Analyzujem taktické rozostavenia... ({i+1}/{total})")
        teams = {}

        for p in f["players"]:
            teams.setdefault(str(p["teamId"]), []).append(copy.deepcopy(p))

        if team_id:
            teams = {tid: players for tid, players in teams.items() if tid == str(team_id)}

        mapped = {}
        teams_meta = {}
        cur_form = {}
        cur_assign = {}
        raw_positions = {}

        for tid, players in teams.items():
            gk = find_goalkeeper(players)
            gk_bottom = True if gk is None else (gk["y"] > PITCH_HEIGHT / 2)

            mapped_players = map_team_visual_band(players, gk_bottom)
            mapped[tid] = mapped_players

            field = [p for p in mapped_players if not p.get("goalkeeper")]
            formation, assign = detect_formation_assignment(field, gk_bottom)

            attackers = [p for p in players if assign.get(p["shirt"]) == "att"]
            wing = detect_wing(attackers, gk_bottom)

            teams_meta[tid] = {
                "name": teams_info[tid]["name"],
                "formation": formation,
                "wing": wing
            }

            cur_form[tid] = formation
            cur_assign[tid] = assign

            raw_positions[tid] = [
                {"shirt": p["shirt"], "x": p["x"], "y": p["y"]}
                for p in players if not p.get("goalkeeper")
            ]

        if i == 0:
            img = draw_snapshot(f, mapped, teams_meta, color_map, "Začiatočná taktika")
            images.append(img)
            prev_form = cur_form
            prev_assign = cur_assign
            prev_positions = raw_positions
            continue

        change = False
        change_lines = []

        for tid in mapped:
            prev = prev_positions.get(tid, [])
            cur = raw_positions.get(tid, [])
            mapping, dists = hungarian_assign(prev, cur)

            if not mapping:
                continue

            stable = 0
            common = set(prev_assign[tid]) & set(cur_assign[tid]) & set(mapping)

            for s in common:
                cur_s = mapping[s]

                if (
                    dists.get(s, 1e6) <= EUCLIDEAN_MATCH_THRESHOLD
                    and prev_assign[tid][s] == cur_assign[tid][cur_s]
                ):
                    stable += 1

            prop = stable / len(common) if common else 0

            if prop < LINE_STABILITY_THRESHOLD or cur_form[tid] != prev_form[tid]:
                if f["clock"] - last_change_clock > MIN_CHANGE_GAP:
                    change = True
                    change_lines.append(f"{teams_meta[tid]['name']} Formácia {prev_form[tid]} → {cur_form[tid]}")

        if change:
            title = f"Zmena taktiky {frame_time(f['clock'])}"
            img = draw_snapshot(f, mapped, teams_meta, color_map, title, " | ".join(change_lines))
            images.append(img)

            prev_form = cur_form
            prev_assign = cur_assign
            prev_positions = raw_positions
            last_change_clock = f["clock"]

    update(status_callback, "done", "✅ Analýza dokončená")
    return images
