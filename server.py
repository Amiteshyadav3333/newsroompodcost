import os
import uuid
import json

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from livekit import api

load_dotenv()  # .env ya .env.local se LIVEKIT_* values padhta hai
load_dotenv(".env.local")

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

# Agar agent.py me agent_name set kiya hai (explicit dispatch), to yahan wahi naam do.
AGENT_NAME = os.environ.get("AGENT_NAME", "")

app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.get("/token")
def token():
    topic = request.args.get("topic", "").strip()[:300]
    prompt = request.args.get("prompt", "").strip()[:1000]
    try:
        speakers = json.loads(request.args.get("speakers", "[]"))
        speakers = [
            {"name": str(item.get("name", ""))[:60], "role": str(item.get("role", ""))[:100]}
            for item in speakers[:4] if isinstance(item, dict)
        ]
    except (ValueError, TypeError):
        speakers = []
    name = json.dumps({"topic": topic, "prompt": prompt, "speakers": speakers}, ensure_ascii=False)
    room = request.args.get("room") or f"room-{uuid.uuid4().hex[:8]}"
    identity = f"user-{uuid.uuid4().hex[:6]}"

    at = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(name)
        .with_grants(api.VideoGrants(room_join=True, room=room))
    )

    if AGENT_NAME:
        at = at.with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)]
            )
        )

    return jsonify({"token": at.to_jwt(), "url": LIVEKIT_URL, "room": room})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
