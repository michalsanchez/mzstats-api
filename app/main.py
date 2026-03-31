from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

print("=== APP MAIN LOADED FOR REAL ===")

app = FastAPI()

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

@app.get("/")
def root():
    return {"status": "ok", "source": "REAL APP MAIN"}

@app.get("/status")
def status():
    return {"step": "idle", "message": "REAL STATUS WORKS"}

@app.get("/headers-debug")
def headers_debug(request: Request):
    return {
        "ok": True,
        "origin": request.headers.get("origin"),
        "host": request.headers.get("host"),
    }
