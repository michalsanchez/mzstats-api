from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import json
import os

from app.data_pipeline.mz_match_to_json import run_pipeline
from app.services.mz_lookup import get_team_match_history, resolve_soccer_team
from app.services.tactical_engine import run_analysis, run_heatmaps_v2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://match.mzstats.app",
        "https://www.match.mzstats.app",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/reports", StaticFiles(directory="reports"), name="reports")

LAST = []

STATUS = {
    "step": "idle",
    "message": "Pripravené",
}


def set_status(step: str, message: str):
    global STATUS
    STATUS = {"step": step, "message": message}


def reset_status():
    set_status("idle", "Pripravené")


def ensure_match_json(match_id: str):
    path = f"data_d_and_p/match_{match_id}.json"

    if not os.path.exists(path):
        run_pipeline(match_id, set_status)
    else:
        set_status("exists", "Zápas už existuje, preskakujem sťahovanie...")

    return path


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/analyzer")


@app.get("/analyzer", include_in_schema=False)
def analyzer_page():
    return RedirectResponse(url="https://match.mzstats.app/analyzer.html")


@app.get("/status")
def get_status():
    return STATUS


@app.get("/resolve-team")
def resolve_team(username: str = None, team_id: str = None):
    try:
        set_status("resolve_team", "Vyhľadávam tím...")
        team = resolve_soccer_team(username=username, team_id=team_id)
        set_status("team_ready", "Tím nájdený")
        return team
    except Exception as e:
        set_status("error", f"Chyba: {str(e)}")
        return {"error": str(e)}


@app.get("/match-history")
def match_history(team_id: str):
    try:
        set_status("match_history", "Načítavam históriu zápasov...")
        matches = get_team_match_history(team_id)
        set_status("history_ready", "História zápasov načítaná")
        return {"matches": matches}
    except Exception as e:
        set_status("error", f"Chyba: {str(e)}")
        return {"matches": [], "error": str(e)}


@app.get("/match-teams")
def match_teams(match_id: str):
    try:
        set_status("prepare_match", "Pripravujem zápas...")
        path = ensure_match_json(match_id)
        set_status("load_teams", "Načítavam tímy a hráčov...")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        set_status("teams_ready", "Tímy načítané")
        return data.get("teams", {})
    except Exception as e:
        set_status("error", f"Chyba: {str(e)}")
        return {"error": str(e)}


@app.get("/analyze")
def analyze(match_id: str, team_id: str = None, full_pitch: bool = False):
    try:
        full_pitch = bool(team_id)
        set_status("prepare_match", "Pripravujem zápas...")
        ensure_match_json(match_id)
        set_status("analysis_start", "Spúšťam taktickú analýzu...")

        imgs = run_analysis(
            match_id,
            team_id=team_id,
            full_pitch=full_pitch,
            status_callback=set_status,
        )

        urls = [f"/reports/tactics/{os.path.basename(p)}" for p in imgs]

        global LAST
        LAST = urls
        set_status("done", "Analýza dokončená")
        return {"images": urls}
    except Exception as e:
        set_status("error", f"Chyba: {str(e)}")
        return {"images": [], "error": str(e)}


@app.get("/heatmaps")
def heatmaps(match_id: str, team_id: str = None, mode: str = "per_player"):
    try:
        set_status("prepare_match", "Pripravujem zápas pre heatmapy...")
        ensure_match_json(match_id)
        set_status("heatmaps_start", "Spúšťam generovanie heatmáp hráčov...")

        imgs = run_heatmaps_v2(
            match_id,
            team_id=team_id,
            mode=mode,
            status_callback=set_status,
        )

        urls = [f"/reports/heatmaps/{os.path.basename(p)}" for p in imgs]

        global LAST
        LAST = urls
        set_status("done", "Heatmapy hráčov dokončené")
        return {"images": urls}
    except Exception as e:
        set_status("error", f"Chyba: {str(e)}")
        return {"images": [], "error": str(e)}
