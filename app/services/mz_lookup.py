import json
import requests
import xml.etree.ElementTree as ET

BASE_URL = "https://www.managerzone.com/"

with open("cookies.json", "r", encoding="utf-8") as f:
    cookies = json.load(f)

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Encoding": "gzip, deflate"
}


# ------------------------------------------------
# TEAM RESOLVE
# ------------------------------------------------

def resolve_soccer_team(username: str = None, team_id: str = None):
    if not username and not team_id:
        raise ValueError("Musíš zadať username alebo team_id")

    params = {"sport_id": 1}

    if username:
        params["username"] = username

    if team_id:
        params["team_id"] = team_id

    r = requests.get(
        BASE_URL + "xml/manager_data.php",
        params=params,
        cookies=cookies,
        headers=headers,
        timeout=30
    )

    r.raise_for_status()

    root = ET.fromstring(r.content)

    user_node = root.find("UserData")

    if user_node is None:
        raise Exception("Nepodarilo sa načítať používateľské dáta")

    resolved_username = user_node.get("username")
    user_id = user_node.get("userId")

    for team in user_node.findall("Team"):
        if team.get("sport") == "soccer":
            return {
                "username": resolved_username,
                "user_id": user_id,
                "team_id": team.get("teamId"),
                "team_name": team.get("teamName"),
                "short_name": team.get("nameShort"),
                "series_name": team.get("seriesName"),
                "series_id": team.get("seriesId"),
                "rank_pos": team.get("rankPos"),
                "rank_points": team.get("rankPoints")
            }

    raise Exception("Nenašiel sa soccer tím pre zadaného používateľa")


# ------------------------------------------------
# MATCH HISTORY
# ------------------------------------------------

def get_team_match_history(team_id: str, limit: int = 100):
    params = {
        "sport_id": 1,
        "team_id": team_id,
        "match_status": 1,
        "limit": limit
    }

    r = requests.get(
        BASE_URL + "xml/team_matchlist.php",
        params=params,
        cookies=cookies,
        headers=headers,
        timeout=30
    )

    r.raise_for_status()

    root = ET.fromstring(r.content)

    matches = []

    for match in root.findall("Match"):
        match_id = match.get("id")
        date = match.get("date")
        type_name = match.get("typeName")
        match_type = match.get("type")

        home = None
        away = None

        for team in match.findall("Team"):
            if team.get("field") == "home":
                home = {
                    "team_id": team.get("teamId"),
                    "team_name": team.get("teamName"),
                    "goals": team.get("goals"),
                    "country": team.get("countryShortname")
                }

            elif team.get("field") == "away":
                away = {
                    "team_id": team.get("teamId"),
                    "team_name": team.get("teamName"),
                    "goals": team.get("goals"),
                    "country": team.get("countryShortname")
                }

        matches.append({
            "match_id": match_id,
            "date": date,
            "type_name": type_name,
            "match_type": match_type,
            "home": home,
            "away": away
        })

    return matches
