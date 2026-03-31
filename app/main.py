from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json

print("=== APP MAIN LOADED FOR REAL ===")

from app.services.tactical_engine import run_analysis
from app.data_pipeline.mz_match_to_json import run_pipeline
from app.services.mz_lookup import resolve_soccer_team, get_team_match_history

app = FastAPI()

# ------------------------------------------------
# CORS
# ------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://match.mzstats.app",
        "https://www.match.mzstats.app",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# STATIC FILES
# ------------------------------------------------

os.makedirs("reports/tactics", exist_ok=True)
os.makedirs("data_d_and_p", exist_ok=True)

app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# ------------------------------------------------
# GLOBAL STATE
# ------------------------------------------------

LAST = []

STATUS = {
    "step": "idle",
    "message": "Pripravené"
}

# ------------------------------------------------
# HELPERS
# ------------------------------------------------

def set_status(step: str, message: str):
    global STATUS
    STATUS = {
        "step": step,
        "message": message
    }

def reset_status():
    set_status("idle", "Pripravené")

def ensure_match_json(match_id: str):
    path = f"data_d_and_p/match_{match_id}.json"

    if not os.path.exists(path):
        set_status("pipeline", "📦 Sťahujem dáta zápasu...")
        run_pipeline(match_id, set_status)
    else:
        set_status("exists", "📂 Zápas už existuje")

    return path

# ------------------------------------------------
# ROOT
# ------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "source": "REAL APP MAIN"}

# ------------------------------------------------
# DEBUG
# ------------------------------------------------

@app.get("/headers-debug")
def headers_debug(request: Request):
    return {
        "ok": True,
        "origin": request.headers.get("origin"),
        "host": request.headers.get("host"),
    }

# ------------------------------------------------
# STATUS
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
        set_status("error", f"❌ {str(e)}")
        return {"error": str(e)}

# ------------------------------------------------
# MATCH HISTORY
# ------------------------------------------------

@app.get("/match-history")
def match_history(team_id: str):
    try:
        set_status("match_history", "📋 Načítavam históriu...")

        matches = get_team_match_history(team_id)

        set_status("history_ready", "✅ História načítaná")
        return {"matches": matches}

    except Exception as e:
        set_status("error", f"❌ {str(e)}")
        return {"matches": [], "error": str(e)}

# ------------------------------------------------
# MATCH TEAMS
# ------------------------------------------------

@app.get("/match-teams")
def match_teams(match_id: str):
    try:
        set_status("prepare_match", "⚽ Pripravujem zápas...")

        path = ensure_match_json(match_id)

        set_status("load_teams", "👥 Načítavam tímy...")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        set_status("teams_ready", "✅ Tímy načítané")

        return data.get("teams", {})

    except Exception as e:
        set_status("error", f"❌ {str(e)}")
        return {"error": str(e)}

# ------------------------------------------------
# ANALYZE
# ------------------------------------------------

@app.get("/analyze")
def analyze(match_id: str, team_id: str = None, full_pitch: bool = False):
    try:
        set_status("prepare_match", "⚽ Pripravujem zápas...")

        ensure_match_json(match_id)

        set_status("analysis_start", "⚙️ Spúšťam analýzu...")

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
        set_status("error", f"❌ {str(e)}")
        return {"images": [], "error": str(e)}
