import requests
import json
import struct
import time
import xml.etree.ElementTree as ET
import os

from playwright.sync_api import sync_playwright

BASE_URL = "https://www.managerzone.com/"

with open("cookies.json") as f:
    cookies = json.load(f)

headers = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
    "Accept-Encoding": "gzip, deflate"
}


# =============================
# STATUS HELPER
# =============================

def update(status_callback, step, message):
    if status_callback:
        status_callback(step, message)


# =============================
# 1️⃣ Inicializácia replay
# =============================

def initialize_replay(match_id, status_callback=None):

    update(status_callback, "init_replay", "⚽ Inicializujem Match Viewer...")

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--single-process",
                "--no-zygote",
            ]
        )

        context = browser.new_context()
        page = context.new_page()

        url = f"https://www.managerzone.com/?p=match&sub=result&type=2d&play=2d&mid={match_id}"

        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        time.sleep(10)

        browser.close()


# =============================
# 2️⃣ Download replay + XML
# =============================

def download_files(match_id, status_callback=None):

    update(status_callback, "download_replay", "📥 Sťahujem replay zápasu...")

    params = {
        "type": "data",
        "mid": match_id,
        "sport": "soccer"
    }

    r = requests.get(
        BASE_URL + "matchviewer/getMatchFiles.php",
        params=params,
        cookies=cookies,
        headers=headers
    )

    if len(r.content) < 100:
        raise Exception("Replay sa nepodarilo stiahnuť")

    replay = r.content

    update(status_callback, "download_xml", "📄 Sťahujem dáta hráčov (XML)...")

    params = {
        "type": "stats",
        "mid": match_id,
        "sport": "soccer"
    }

    r = requests.get(
        BASE_URL + "matchviewer/getMatchFiles.php",
        params=params,
        cookies=cookies,
        headers=headers
    )

    if len(r.content) < 100:
        raise Exception("XML sa nepodarilo stiahnuť")

    xml_data = r.content

    return replay, xml_data


# =============================
# 3️⃣ Parse XML hráčov
# =============================

def parse_players(xml_data, status_callback=None):

    update(status_callback, "parse_players", "👥 Načítavam tímy a hráčov...")

    root = ET.fromstring(xml_data)

    players_map = {}
    teams = {}

    for team in root.findall("Team"):

        team_id = team.get("id")

        teams[team_id] = {
            "name": team.get("name"),
            "visiting": team.get("visiting")
        }

    for player in root.findall("Player"):

        internal_id = int(player.get("internalId"))

        players_map[internal_id] = {
            "playerId": player.get("id"),
            "teamId": player.get("teamId"),
            "name": player.get("name"),
            "shirt": player.get("shirtno"),
            "goalkeeper": player.get("goalkeeper") == "1"
        }

    return players_map, teams


# =============================
# 4️⃣ Parse replay BIN
# =============================

def parse_replay(bin_data, players_map, status_callback=None):

    update(status_callback, "parse_replay", "🧩 Parsujem replay dáta...")

    offset = 0

    match_id = struct.unpack("<I", bin_data[offset:offset+4])[0]
    offset += 4

    num_actors = struct.unpack("<I", bin_data[offset:offset+4])[0]
    offset += 4

    secs_per_frame = struct.unpack("<f", bin_data[offset:offset+4])[0]
    offset += 4

    frame_size = 10 + num_actors * 7
    total_frames = (len(bin_data) - offset) // frame_size

    frames = []

    for frame_no in range(total_frames):

        frame_offset = offset + frame_no * frame_size
        cursor = frame_offset

        clock = struct.unpack("<i", bin_data[cursor:cursor+4])[0]
        cursor += 4

        ball_x = struct.unpack("<h", bin_data[cursor:cursor+2])[0]
        cursor += 2

        ball_y = struct.unpack("<h", bin_data[cursor:cursor+2])[0]
        cursor += 2

        ball_z = struct.unpack("<h", bin_data[cursor:cursor+2])[0]
        cursor += 2

        players = []

        for i in range(num_actors):

            internal_id = bin_data[cursor]
            cursor += 1

            status = bin_data[cursor]
            cursor += 1

            x = struct.unpack("<h", bin_data[cursor:cursor+2])[0]
            cursor += 2

            y = struct.unpack("<h", bin_data[cursor:cursor+2])[0]
            cursor += 2

            d = bin_data[cursor]
            cursor += 1

            if internal_id != 0:

                info = players_map.get(internal_id, {})

                players.append({
                    "internalId": internal_id,
                    "name": info.get("name"),
                    "teamId": info.get("teamId"),
                    "shirt": info.get("shirt"),
                    "goalkeeper": info.get("goalkeeper"),
                    "status": status,
                    "x": x,
                    "y": y,
                    "d": d
                })

        frames.append({
            "frame": frame_no,
            "clock": clock,
            "ball": {
                "x": ball_x,
                "y": ball_y,
                "z": ball_z
            },
            "players": players
        })

    return {
        "matchId": match_id,
        "numActors": num_actors,
        "secsPerFrame": secs_per_frame,
        "totalFrames": total_frames,
        "frames": frames
    }


# =============================
# PIPELINE
# =============================

def run_pipeline(match_id, status_callback=None):

    os.makedirs("data_d_and_p", exist_ok=True)

    output_path = f"data_d_and_p/match_{match_id}.json"

    if os.path.exists(output_path):
        update(status_callback, "exists", "📂 Zápas už existuje, preskakujem sťahovanie...")
        return output_path

    initialize_replay(match_id, status_callback)

    replay, xml_data = download_files(match_id, status_callback)

    players_map, teams = parse_players(xml_data, status_callback)

    match_data = parse_replay(replay, players_map, status_callback)

    match_data["teams"] = teams
    match_data["players"] = players_map

    update(status_callback, "save", "💾 Ukladám zápasové dáta...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(match_data, f, ensure_ascii=False)

    update(status_callback, "done", "✅ Zápas pripravený")

    return output_path


if __name__ == "__main__":
    run_pipeline("1582082965")
