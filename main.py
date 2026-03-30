from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from analyzer import run_analysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/status")
def status():
    return {"step": "ready", "message": "Ready"}

@app.get("/analyze")
def analyze(match_id: str = "0", team_id: str = None, full_pitch: bool = False):
    try:
        images = run_analysis(match_id, team_id=team_id, full_pitch=full_pitch)
        return {
            "match_id": match_id,
            "images": images,
            "message": "Analyze endpoint funguje"
        }
    except Exception as e:
        return {
            "error": str(e),
            "images": []
        }
