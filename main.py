from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/status")
def status():
    return {"step": "ready", "message": "Ready"}