from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os
import uvicorn

# APP SETUP

load_dotenv()

app = FastAPI(
    title="MYLO AGENT",
    version="0.0.1"
)

uvicorn.run(app, host=os.getenv("SERVER_IP"), port=int(os.getenv("SERVER_PORT")))

# ROUTES
@app.get("/")
async def root():
    return {"message": "MYLO AGENT is running!"}