from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import json

from app.services.tactical_engine import run_analysis
from app.data_pipeline.mz_match_to_json import run_pipeline
from app.services.mz_lookup import resolve_soccer_team, get_team_match_history

app = FastAPI()

app.mount("/reports", StaticFiles(directory="reports"), name="reports")

LAST = []

STATUS = {
    "step": "idle",
    "message": "Pripravené"
}


# ------------------------------------------------
# STATUS HELPERS
# ------------------------------------------------

def set_status(step: str, message: str):
    global STATUS
    STATUS = {
        "step": step,
        "message": message
    }


def reset_status():
    set_status("idle", "Pripravené")


# ------------------------------------------------
# MATCH JSON ENSURE
# ------------------------------------------------

def ensure_match_json(match_id: str):
    path = f"data_d_and_p/match_{match_id}.json"

    if not os.path.exists(path):
        run_pipeline(match_id, set_status)
    else:
        set_status("exists", "📂 Zápas už existuje, preskakujem sťahovanie...")

    return path


# ------------------------------------------------
# UI PAGE
# ------------------------------------------------

@app.get("/analyzer", response_class=HTMLResponse)
def analyzer_page():
    with open("analyzer.html", "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------
# LIVE STATUS API
# ------------------------------------------------

@app.get("/status")
def get_status():
    return STATUS


# ------------------------------------------------
# TEAM LOOKUP
# ------------------------------------------------

@app.get("/resolve-team")
def resolve_team(username: str = None, team_id: str = None):
    try:
        set_status("resolve_team", "🔎 Vyhľadávam tím...")
        team = resolve_soccer_team(username=username, team_id=team_id)
        set_status("team_ready", "✅ Tím nájdený")
        return team
    except Exception as e:
        set_status("error", f"❌ Chyba: {str(e)}")
        return {"error": str(e)}


@app.get("/match-history")
def match_history(team_id: str):
    try:
        set_status("match_history", "📋 Načítavam históriu zápasov...")
        matches = get_team_match_history(team_id)
        set_status("history_ready", "✅ História zápasov načítaná")
        return {"matches": matches}
    except Exception as e:
        set_status("error", f"❌ Chyba: {str(e)}")
        return {"matches": [], "error": str(e)}


# ------------------------------------------------
# LOAD MATCH TEAMS
# ------------------------------------------------

@app.get("/match-teams")
def match_teams(match_id: str):
    try:
        set_status("prepare_match", "⚽ Pripravujem zápas...")

        path = ensure_match_json(match_id)

        set_status("load_teams", "👥 Načítavam tímy a hráčov...")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        set_status("teams_ready", "✅ Tímy načítané")

        return data.get("teams", {})

    except Exception as e:
        set_status("error", f"❌ Chyba: {str(e)}")
        return {"error": str(e)}


# ------------------------------------------------
# MAIN ANALYSIS
# ------------------------------------------------

@app.get("/analyze")
def analyze(match_id: str, team_id: str = None, full_pitch: bool = False):
    try:
        set_status("prepare_match", "⚽ Pripravujem zápas...")

        ensure_match_json(match_id)

        set_status("analysis_start", "⚙️ Spúšťam taktickú analýzu...")

        imgs = run_analysis(
            match_id,
            team_id=team_id,
            full_pitch=full_pitch,
            status_callback=set_status
        )

        urls = []

        for p in imgs:
            fn = os.path.basename(p)
            urls.append(f"/reports/tactics/{fn}")

        global LAST
        LAST = urls

        set_status("done", "✅ Analýza dokončená")

        return {"images": urls}

    except Exception as e:
        set_status("error", f"❌ Chyba: {str(e)}")
        return {"images": [], "error": str(e)}
