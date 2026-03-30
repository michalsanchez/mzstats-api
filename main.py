from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
def analyze(match_id: int = 0):
    return {
        "match_id": match_id,
        "images": [],
        "message": "Analyze endpoint funguje"
    }
