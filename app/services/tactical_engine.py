import os
import copy
import math
import json
import textwrap
import unicodedata
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

PITCH_WIDTH = 750
PITCH_HEIGHT = 1000
SECTOR_HEIGHT = PITCH_HEIGHT / 10
SNAPSHOT_SCALE = 3 / 7

PENALTY_DEPTH = int(PITCH_HEIGHT * 0.12)
CENTER_RADIUS = int(PITCH_HEIGHT * 0.12)

HALF_TIME = 45 * 60

EUCLIDEAN_MATCH_THRESHOLD = 140.0
LINE_STABILITY_THRESHOLD = 0.85

DEBOUNCE_REQUIRED = 3
MIN_CHANGE_GAP = 300

BASE_COLORS = ["#d7263d", "#0b66c3"]

HEATMAP_LEVELS = [
    (0.1, None, 0.0),
    (0.3, "#facc15", 0.3),
    (0.45, "#f97316", 0.6),
    (0.65, "#f97316", 0.8),
    (0.8, "#dc2626", 0.9),
    (1.01, "#dc2626", 1.0),
]

HEATMAP_LEVELS_COOL = [
    (0.1, None, 0.0),
    (0.3, "#67e8f9", 0.28),
    (0.45, "#38bdf8", 0.5),
    (0.65, "#2563eb", 0.7),
    (0.8, "#1d4ed8", 0.86),
    (1.01, "#1e3a8a", 1.0),
]

KICK_POINTS = [
    (292, 55), (453, 55), (292, 946), (453, 946)
]

TOL = 5


# ------------------------------------------------
# STATUS HELPER
# ------------------------------------------------

def update(status_callback, step, message):
    if status_callback:
        status_callback(step, message)


# ------------------------------------------------
# HELPERS
# ------------------------------------------------

def near(a, b):
    return abs(a - b) <= TOL


def load_match(match_id):
    path = f"data_d_and_p/match_{match_id}.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def frame_time(clock):
    s = int(clock)
    return f"{s//60:02d}:{s%60:02d}"


def safe_name(value, fallback="player"):
    text = unicodedata.normalize("NFKD", str(value or fallback))
    text = text.encode("ascii", "ignore").decode("ascii")
    cleaned = [
        ch.lower() if ch.isalnum() else "_"
        for ch in text
    ]
    return "".join(cleaned).strip("_") or fallback


def build_goalkeeper_reference(frames):
    reference = {}

    for frame in frames:
        by_team = defaultdict(list)
        for player in frame["players"]:
            by_team[str(player.get("teamId"))].append(player)

        for team_id, players in by_team.items():
            if team_id in reference:
                continue
            gk = find_goalkeeper(players)
            if gk is not None:
                reference[team_id] = gk["y"] > PITCH_HEIGHT / 2

        if reference:
            missing = [team_id for team_id in by_team if team_id not in reference]
            if not missing:
                break

    return reference


def normalize_heatmap_point(x, y, current_gk_bottom, reference_gk_bottom):
    if current_gk_bottom is None or reference_gk_bottom is None:
        return x, y
    if current_gk_bottom != reference_gk_bottom:
        return PITCH_WIDTH - x, PITCH_HEIGHT - y
    return x, y


# ------------------------------------------------
# GOAL KICK DETECTION
# ------------------------------------------------

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


# ------------------------------------------------
# TEAM HELPERS
# ------------------------------------------------

def find_goalkeeper(players):
    for p in players:
        if p.get("goalkeeper"):
            return p
    return None


# ------------------------------------------------
# POSITION NORMALIZATION
# ------------------------------------------------

def map_team_visual_band(players, gk_bottom, full_pitch=False):
    pts = [dict(p) for p in players]

    field = [p for p in pts if not p.get("goalkeeper")]

    if full_pitch:
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

        inner_top = SECTOR_HEIGHT * 2
        inner_bottom = PITCH_HEIGHT - (SECTOR_HEIGHT * 2)

        for p in pts:
            p["display_x"] = p["x"]

            if p.get("goalkeeper"):
                p["display_y"] = (
                    inner_bottom if gk_bottom else inner_top
                )
            else:
                p["display_y"] = inner_top + (
                    (p["y"] - miny) / (maxy - miny)
                ) * (inner_bottom - inner_top)
        return pts

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


# ------------------------------------------------
# FORMATION DETECTION
# ------------------------------------------------

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


# ------------------------------------------------
# PLAYER TRACKING
# ------------------------------------------------

def hungarian_assign(prev_players, cur_players):
    if not prev_players or not cur_players:
        return {}, {}

    n = max(len(prev_players), len(cur_players))
    cost = np.full((n, n), 1e6)

    for i, p in enumerate(prev_players):
        for j, q in enumerate(cur_players):
            cost[i, j] = math.hypot(
                p["x"] - q["x"],
                p["y"] - q["y"]
            )

    r, c = linear_sum_assignment(cost)

    mapping = {}
    dists = {}

    for i, j in zip(r, c):
        if i < len(prev_players) and j < len(cur_players):
            mapping[prev_players[i]["shirt"]] = cur_players[j]["shirt"]
            dists[prev_players[i]["shirt"]] = cost[i, j]

    return mapping, dists


# ------------------------------------------------
# WING DETECTION
# ------------------------------------------------

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


# ------------------------------------------------
# DRAWING
# ------------------------------------------------

def draw_pitch(ax):
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            PITCH_WIDTH,
            PITCH_HEIGHT,
            facecolor="#4caf50"
        )
    )

    stripe = PITCH_HEIGHT / 10

    for i in range(10):
        if i % 2 == 0:
            ax.add_patch(
                patches.Rectangle(
                    (0, i * stripe),
                    PITCH_WIDTH,
                    stripe,
                    facecolor="#43a047"
                )
            )

    ax.plot(
        [0, PITCH_WIDTH],
        [PITCH_HEIGHT / 2, PITCH_HEIGHT / 2],
        color="white",
        linewidth=2 * SNAPSHOT_SCALE
    )

    ax.add_patch(
        patches.Circle(
            (PITCH_WIDTH / 2, PITCH_HEIGHT / 2),
            CENTER_RADIUS,
            fill=False,
            edgecolor="white",
            linewidth=2 * SNAPSHOT_SCALE
        )
    )

    ax.add_patch(
        patches.Rectangle(
            (PITCH_WIDTH * 0.2, 0),
            PITCH_WIDTH * 0.6,
            PENALTY_DEPTH,
            fill=False,
            edgecolor="white",
            linewidth=2 * SNAPSHOT_SCALE
        )
    )

    ax.add_patch(
        patches.Rectangle(
            (PITCH_WIDTH * 0.2, PITCH_HEIGHT - PENALTY_DEPTH),
            PITCH_WIDTH * 0.6,
            PENALTY_DEPTH,
            fill=False,
            edgecolor="white",
            linewidth=2 * SNAPSHOT_SCALE
        )
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
        fontsize=11 * SNAPSHOT_SCALE,
        fontweight="bold",
        color="#b00000",
        bbox=dict(
            facecolor="#ffeaea",
            edgecolor="#cc0000",
            boxstyle=f"round,pad={0.6 * SNAPSHOT_SCALE}"
        )
    )


def draw_snapshot(frame, teams, teams_meta, color_map, title, change_text=None):
    os.makedirs("reports/tactics", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5 * SNAPSHOT_SCALE, 11 * SNAPSHOT_SCALE))
    plt.subplots_adjust(top=0.90)

    ax.set_xlim(0, PITCH_WIDTH)
    ax.set_ylim(0, PITCH_HEIGHT)
    ax.axis("off")

    fig.suptitle(title, fontsize=16 * SNAPSHOT_SCALE, y=0.985)

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
            fontsize=12 * SNAPSHOT_SCALE,
            fontweight="bold",
            bbox=dict(
                facecolor=color,
                alpha=0.18,
                boxstyle=f"round,pad={0.5 * SNAPSHOT_SCALE}"
            )
        )

        for p in players:
            if p.get("goalkeeper"):
                continue

            ax.scatter(
                p["display_x"],
                p["display_y"],
                c=color,
                s=160 * SNAPSHOT_SCALE,
                edgecolors="black",
                linewidths=SNAPSHOT_SCALE
            )

            ax.text(
                p["display_x"],
                p["display_y"] - (120 * SNAPSHOT_SCALE),
                str(p["shirt"]),
                ha="center",
                fontsize=28 * SNAPSHOT_SCALE * 0.7,
                fontweight="bold",
                path_effects=[
                    pe.withStroke(linewidth=1, foreground=color)
                ]
            )

    out = f"reports/tactics/frame_{frame['frame']}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()

    return out


def collect_player_tracks(frames, team_filter=None):
    tracks = {}
    reference_gk_by_team = build_goalkeeper_reference(frames)

    for frame in frames:
        by_team = defaultdict(list)
        for player in frame["players"]:
            by_team[str(player.get("teamId"))].append(player)

        current_gk_by_team = {}
        for team_id, players in by_team.items():
            gk = find_goalkeeper(players)
            current_gk_by_team[team_id] = None if gk is None else (gk["y"] > PITCH_HEIGHT / 2)

        for player in frame["players"]:
            if player.get("goalkeeper"):
                continue

            tid = str(player.get("teamId"))

            if team_filter and tid != str(team_filter):
                continue

            key = (tid, str(player.get("shirt")))

            if key not in tracks:
                tracks[key] = {
                    "team_id": tid,
                    "shirt": str(player.get("shirt")),
                    "name": player.get("name") or f"Player {player.get('shirt')}",
                    "points": []
                }

            x, y = normalize_heatmap_point(
                player["x"],
                player["y"],
                current_gk_by_team.get(tid),
                reference_gk_by_team.get(tid),
            )
            tracks[key]["points"].append((x, y))

    return tracks


def draw_player_heatmap(player_track, team_name, color, output_dir):
    xs = [p[0] for p in player_track["points"]]
    ys = [p[1] for p in player_track["points"]]

    fig, ax = plt.subplots(figsize=(8.5 * SNAPSHOT_SCALE, 11 * SNAPSHOT_SCALE))
    plt.subplots_adjust(top=0.88)

    ax.set_xlim(0, PITCH_WIDTH)
    ax.set_ylim(0, PITCH_HEIGHT)
    ax.axis("off")

    draw_pitch(ax)

    hist, xedges, yedges = np.histogram2d(
        xs,
        ys,
        bins=(22, 30),
        range=[[0, PITCH_WIDTH], [0, PITCH_HEIGHT]]
    )

    hist = hist.T
    max_value = np.max(hist)

    if max_value > 0:
        for row in range(hist.shape[0]):
            for col in range(hist.shape[1]):
                value = hist[row, col]

                if value <= 0:
                    continue

                ratio = value / max_value
                fill_color = None
                fill_alpha = 0.0

                for threshold, level_color, level_alpha in HEATMAP_LEVELS:
                    if ratio <= threshold:
                        fill_color = level_color
                        fill_alpha = level_alpha
                        break

                if not fill_color or fill_alpha <= 0:
                    continue

                x0 = xedges[col]
                x1 = xedges[col + 1]
                y0 = yedges[row]
                y1 = yedges[row + 1]

                ax.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        facecolor=fill_color,
                        edgecolor="none",
                        alpha=fill_alpha,
                        zorder=2
                    )
                )

    title = f"{team_name} | #{player_track['shirt']} {player_track['name']}"

    fig.suptitle(title, fontsize=16 * SNAPSHOT_SCALE, y=0.97)

    ax.text(
        PITCH_WIDTH / 2,
        PITCH_HEIGHT * 0.04,
        f"Pohyby hráča | {len(player_track['points'])} záznamov",
        ha="center",
        fontsize=11 * SNAPSHOT_SCALE,
        fontweight="bold",
        color="white",
        bbox=dict(
            facecolor=color,
            alpha=0.22,
            boxstyle=f"round,pad={0.45 * SNAPSHOT_SCALE}"
        )
    )

    filename = (
        f"{player_track['team_id']}_"
        f"{player_track['shirt']}_"
        f"{safe_name(player_track['name'])}.png"
    )
    out = os.path.join(output_dir, filename)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()

    return out


def draw_heat_cells(ax, hist, xedges, yedges, levels):
    max_value = np.max(hist)

    if max_value <= 0:
        return

    for row in range(hist.shape[0]):
        for col in range(hist.shape[1]):
            value = hist[row, col]

            if value <= 0:
                continue

            ratio = value / max_value
            fill_color = None
            fill_alpha = 0.0

            for threshold, level_color, level_alpha in levels:
                if ratio <= threshold:
                    fill_color = level_color
                    fill_alpha = level_alpha
                    break

            if not fill_color or fill_alpha <= 0:
                continue

            ax.add_patch(
                patches.Rectangle(
                    (xedges[col], yedges[row]),
                    xedges[col + 1] - xedges[col],
                    yedges[row + 1] - yedges[row],
                    facecolor=fill_color,
                    edgecolor="none",
                    alpha=fill_alpha,
                    zorder=2
                )
            )


def build_track_histogram(track_items):
    xs = []
    ys = []

    for item in track_items:
        for x, y in item["points"]:
            xs.append(x)
            ys.append(y)

    if not xs:
        return None, None, None, None, None

    hist, xedges, yedges = np.histogram2d(
        xs,
        ys,
        bins=(22, 30),
        range=[[0, PITCH_WIDTH], [0, PITCH_HEIGHT]]
    )

    return hist.T, xedges, yedges, np.mean(xs), np.mean(ys)


def draw_combined_team_heatmap(track_items, team_name, color, output_dir, filename):
    hist, xedges, yedges, mean_x, mean_y = build_track_histogram(track_items)

    if hist is None:
        return None

    fig, ax = plt.subplots(figsize=(8.5 * SNAPSHOT_SCALE, 11 * SNAPSHOT_SCALE))
    plt.subplots_adjust(top=0.88)
    ax.set_xlim(0, PITCH_WIDTH)
    ax.set_ylim(0, PITCH_HEIGHT)
    ax.axis("off")

    draw_pitch(ax)
    draw_heat_cells(ax, hist, xedges, yedges, HEATMAP_LEVELS)

    ax.scatter(
        mean_x,
        mean_y,
        c=color,
        s=240 * SNAPSHOT_SCALE,
        edgecolors="white",
        linewidths=1.3 * SNAPSHOT_SCALE,
        zorder=4
    )

    fig.suptitle(
        f"{team_name} | všetci hráči",
        fontsize=16 * SNAPSHOT_SCALE,
        y=0.97
    )

    out = os.path.join(output_dir, filename)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return out


def draw_dual_team_heatmap(team_a_tracks, team_b_tracks, team_a_name, team_b_name, output_dir):
    team_a_hist, xedges_a, yedges_a, mean_ax, mean_ay = build_track_histogram(team_a_tracks)
    team_b_hist, xedges_b, yedges_b, mean_bx, mean_by = build_track_histogram(team_b_tracks)

    fig, ax = plt.subplots(figsize=(8.5 * SNAPSHOT_SCALE, 11 * SNAPSHOT_SCALE))
    plt.subplots_adjust(top=0.86)
    ax.set_xlim(0, PITCH_WIDTH)
    ax.set_ylim(0, PITCH_HEIGHT)
    ax.axis("off")

    draw_pitch(ax)

    if team_a_hist is not None:
        draw_heat_cells(ax, team_a_hist, xedges_a, yedges_a, HEATMAP_LEVELS)
        ax.scatter(mean_ax, mean_ay, c=BASE_COLORS[0], s=200 * SNAPSHOT_SCALE, edgecolors="white", linewidths=1.2 * SNAPSHOT_SCALE, zorder=4)

    if team_b_hist is not None:
        draw_heat_cells(ax, team_b_hist, xedges_b, yedges_b, HEATMAP_LEVELS_COOL)
        ax.scatter(mean_bx, mean_by, c=BASE_COLORS[1], s=200 * SNAPSHOT_SCALE, edgecolors="white", linewidths=1.2 * SNAPSHOT_SCALE, zorder=4)

    fig.suptitle("Všetci hráči | obe mužstvá", fontsize=16 * SNAPSHOT_SCALE, y=0.97)

    ax.text(
        PITCH_WIDTH / 2,
        PITCH_HEIGHT * 0.04,
        f"Teplé farby: {team_a_name} | Studené farby: {team_b_name}",
        ha="center",
        fontsize=11 * SNAPSHOT_SCALE,
        fontweight="bold",
        color="white",
        bbox=dict(facecolor="#101820", alpha=0.45, boxstyle=f"round,pad={0.45 * SNAPSHOT_SCALE}")
    )

    out = os.path.join(output_dir, "all_players_dual_team.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return out


def run_heatmaps(match_id, team_id=None, mode="per_player", status_callback=None):
    update(status_callback, "load_match", "🧭 Načítavam dáta pre heatmapy...")

    data = load_match(match_id)
    frames = data["frames"]
    teams_info = data["teams"]

    output_dir = "reports/heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    tracks = collect_player_tracks(frames, team_filter=team_id)

    if not tracks:
        update(status_callback, "done", "⚠️ Nenašli sa žiadni hráči pre heatmapy")
        return []

    team_order = []

    for frame in frames:
        for player in frame["players"]:
            tid = str(player["teamId"])
            if team_id and tid != str(team_id):
                continue
            if tid not in team_order:
                team_order.append(tid)

    color_map = {tid: BASE_COLORS[i % len(BASE_COLORS)] for i, tid in enumerate(team_order)}

    ordered_tracks = sorted(
        tracks.values(),
        key=lambda item: (item["team_id"], int(item["shirt"]) if str(item["shirt"]).isdigit() else 999)
    )

    images = []
    total = len(ordered_tracks)

    for index, track in enumerate(ordered_tracks, start=1):
        update(
            status_callback,
            "heatmaps",
            f"🔥 Generujem heatmapy hráčov... ({index}/{total})"
        )

        team_name = teams_info.get(track["team_id"], {}).get("name", track["team_id"])
        color = color_map.get(track["team_id"], BASE_COLORS[0])
        images.append(draw_player_heatmap(track, team_name, color, output_dir))

    update(status_callback, "done", "✅ Heatmapy hráčov dokončené")
    return images


# ------------------------------------------------
# CLEAN HEATMAP API
# ------------------------------------------------

def run_heatmaps_v2(match_id, team_id=None, mode="per_player", status_callback=None):
    update(status_callback, "load_match", "Načítavam dáta pre heatmapy...")

    data = load_match(match_id)
    frames = data["frames"]
    teams_info = data["teams"]

    output_dir = "reports/heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    all_tracks = collect_player_tracks(frames)
    if not all_tracks:
        update(status_callback, "done", "Nenašli sa žiadni hráči pre heatmapy")
        return []

    filtered_tracks = collect_player_tracks(frames, team_filter=team_id) if team_id else all_tracks
    if not filtered_tracks:
        update(status_callback, "done", "Pre zvolený filter sa nenašli žiadne heatmapy")
        return []

    team_order = []
    for frame in frames:
        for player in frame["players"]:
            tid = str(player["teamId"])
            if tid not in team_order:
                team_order.append(tid)

    color_map = {tid: BASE_COLORS[i % len(BASE_COLORS)] for i, tid in enumerate(team_order)}

    if mode == "combined_all":
        if len(team_order) < 2:
            only_team = team_order[0] if team_order else str(team_id or "")
            only_tracks = sorted(
                collect_player_tracks(frames, team_filter=only_team).values(),
                key=lambda item: int(item["shirt"]) if str(item["shirt"]).isdigit() else 999
            )
            if not only_tracks:
                update(status_callback, "done", "Nenašli sa hráči pre spoločnú heatmapu")
                return []

            team_name = teams_info.get(str(only_team), {}).get("name", str(only_team))
            color = color_map.get(str(only_team), BASE_COLORS[0])
            update(status_callback, "heatmaps", f"Generujem spoločnú heatmapu tímu {team_name}...")
            image = draw_combined_team_heatmap(
                only_tracks,
                team_name,
                color,
                output_dir,
                "all_players_single_team.png",
            )
            update(status_callback, "done", "Spoločná heatmapa je pripravená")
            return [image]

        team_a_id, team_b_id = team_order[:2]
        team_a_tracks = sorted(
            collect_player_tracks(frames, team_filter=team_a_id).values(),
            key=lambda item: int(item["shirt"]) if str(item["shirt"]).isdigit() else 999
        )
        team_b_tracks = sorted(
            collect_player_tracks(frames, team_filter=team_b_id).values(),
            key=lambda item: int(item["shirt"]) if str(item["shirt"]).isdigit() else 999
        )

        if not team_a_tracks and not team_b_tracks:
            update(status_callback, "done", "Nenašli sa hráči pre spoločnú heatmapu")
            return []

        update(status_callback, "heatmaps", "Generujem spoločnú heatmapu oboch tímov...")
        image = draw_dual_team_heatmap(
            team_a_tracks,
            team_b_tracks,
            teams_info.get(team_a_id, {}).get("name", team_a_id),
            teams_info.get(team_b_id, {}).get("name", team_b_id),
            output_dir,
        )
        update(status_callback, "done", "Spoločná heatmapa je pripravená")
        return [image]

    if mode == "combined_team":
        ordered_tracks = sorted(
            filtered_tracks.values(),
            key=lambda item: int(item["shirt"]) if str(item["shirt"]).isdigit() else 999
        )
        if not ordered_tracks:
            update(status_callback, "done", "Nenašli sa hráči pre tímovú heatmapu")
            return []

        team_name = teams_info.get(ordered_tracks[0]["team_id"], {}).get("name", ordered_tracks[0]["team_id"])
        color = color_map.get(ordered_tracks[0]["team_id"], BASE_COLORS[0])
        update(status_callback, "heatmaps", f"Generujem spoločnú heatmapu tímu {team_name}...")
        image = draw_combined_team_heatmap(
            ordered_tracks,
            team_name,
            color,
            output_dir,
            f"all_players_{safe_name(team_name, 'team')}.png",
        )
        update(status_callback, "done", f"Heatmapa tímu {team_name} je pripravená")
        return [image]

    ordered_tracks = sorted(
        filtered_tracks.values(),
        key=lambda item: (item["team_id"], int(item["shirt"]) if str(item["shirt"]).isdigit() else 999)
    )

    images = []
    total = len(ordered_tracks)
    for index, track in enumerate(ordered_tracks, start=1):
        update(status_callback, "heatmaps", f"Generujem heatmapy hráčov... ({index}/{total})")
        team_name = teams_info.get(track["team_id"], {}).get("name", track["team_id"])
        color = color_map.get(track["team_id"], BASE_COLORS[0])
        images.append(draw_player_heatmap(track, team_name, color, output_dir))

    update(status_callback, "done", "Heatmapy hráčov dokončené")
    return images


# ------------------------------------------------
# MAIN ANALYSIS
# ------------------------------------------------

def run_analysis(match_id, team_id=None, full_pitch=False, status_callback=None):
    update(status_callback, "load_match", "Načítavam zápasové dáta...")

    data = load_match(match_id)

    frames = data["frames"]
    teams_info = data["teams"]

    update(status_callback, "freeze_frames", "Vyhľadávam kľúčové herné situácie...")
    freeze_frames = find_freeze_frames(frames)

    if not freeze_frames:
        update(status_callback, "done", "Nenašli sa žiadne taktické situácie")
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
        update(
            status_callback,
            "analyzing",
            f"Analyzujem taktické rozostavenia... ({i+1}/{total})"
        )

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

            mapped_players = map_team_visual_band(
                players,
                gk_bottom,
                full_pitch=full_pitch
            )
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
            update(status_callback, "render_first", "Generujem začiatočný taktický snapshot...")

            img = draw_snapshot(
                f,
                mapped,
                teams_meta,
                color_map,
                "Začiatočná taktika"
            )

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
                    change_lines.append(
                        f"{teams_meta[tid]['name']} Formácia {prev_form[tid]} → {cur_form[tid]}"
                    )

        if change:
            update(status_callback, "render_change", f"Generujem snapshot zmeny taktiky ({frame_time(f['clock'])})...")

            title = f"Zmena taktiky {frame_time(f['clock'])}"

            img = draw_snapshot(
                f,
                mapped,
                teams_meta,
                color_map,
                title,
                " | ".join(change_lines)
            )

            images.append(img)

            prev_form = cur_form
            prev_assign = cur_assign
            prev_positions = raw_positions
            last_change_clock = f["clock"]

    update(status_callback, "done", "Analýza dokončená")

    return images
