"""
WebSocket relay for the paperclip-remote Android app.

Forwards frames verbatim between two peers sharing a room id, where one
peer joins with role=controller and the other with role=controlled.

Run:
    pip install -r requirements.txt
    uvicorn relay:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status

logger = logging.getLogger("relay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROLES = {"controller", "controlled"}
MAX_JOIN_PER_MIN = 10  # per source IP, defense against room-code brute force
ROOM_IDLE_TTL_SEC = 300  # rooms with no peers and no joins for 5 min get reaped


def _client_ip(ws: WebSocket) -> str:
    """Best-effort original-client IP.

    Behind a reverse proxy (Caddy / nginx — see docker-compose.yml
    `tls` profile) the TCP peer is the proxy container, not the phone.
    Honour X-Forwarded-For / X-Real-IP so the rate limiter sees real
    client addresses and one misconfigured app doesn't get the proxy
    permanently throttled.

    Trusting these headers requires that they cannot be spoofed by
    untrusted callers — the relay endpoint is expected to sit behind
    the proxy (no direct internet exposure on port 8765). Self-hosted
    same-owner deployments per SEED §4 satisfy this.
    """
    xff = ws.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",", 1)[0].strip() or "unknown"
    real = ws.headers.get("x-real-ip")
    if real:
        return real
    return ws.client.host if ws.client else "unknown"


@dataclass
class Room:
    controller: WebSocket | None = None
    controlled: WebSocket | None = None
    created_at: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def peer_of(self, role: str) -> WebSocket | None:
        return self.controlled if role == "controller" else self.controller

    def slot(self, role: str) -> WebSocket | None:
        return self.controller if role == "controller" else self.controlled

    def set_slot(self, role: str, ws: WebSocket | None) -> None:
        if role == "controller":
            self.controller = ws
        else:
            self.controlled = ws

    def empty(self) -> bool:
        return self.controller is None and self.controlled is None


rooms: dict[str, Room] = {}
rooms_lock = asyncio.Lock()
joins_by_ip: dict[str, deque[float]] = defaultdict(deque)


async def get_or_create_room(room_id: str) -> Room:
    async with rooms_lock:
        room = rooms.get(room_id)
        if room is None:
            room = Room()
            rooms[room_id] = room
        return room


async def drop_room_if_empty(room_id: str) -> None:
    async with rooms_lock:
        room = rooms.get(room_id)
        if room and room.empty():
            del rooms[room_id]


def rate_limit(ip: str) -> bool:
    """Return False if the caller exceeded MAX_JOIN_PER_MIN in the last 60s."""
    now = time.monotonic()
    q = joins_by_ip[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= MAX_JOIN_PER_MIN:
        return False
    q.append(now)
    return True


app = FastAPI(title="paperclip-remote relay")


@app.get("/healthz")
async def healthz():
    return {"ok": True, "rooms": len(rooms)}


@app.websocket("/ws/{room_id}/{role}")
async def ws_endpoint(ws: WebSocket, room_id: str, role: str):
    client_ip = _client_ip(ws)

    if role not in ROLES:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="invalid role")
        return

    if len(room_id) != 6 or not room_id.isalnum():
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="invalid room")
        return

    if not rate_limit(client_ip):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="rate limited")
        logger.warning("rate limited ip=%s", client_ip)
        return

    room = await get_or_create_room(room_id)

    # Take the slot; kick the previous occupant if any (likely a flapping
    # mobile reconnect — newcomer wins per PROTOCOL.md §endpoint).
    async with room.lock:
        prior = room.slot(role)
        if prior is not None:
            try:
                await prior.close(code=status.WS_1001_GOING_AWAY, reason="replaced")
            except Exception:
                pass
        room.set_slot(role, ws)

    await ws.accept()
    logger.info("join room=%s role=%s ip=%s", room_id, role, client_ip)

    try:
        while True:
            message = await ws.receive()
            mtype = message.get("type")
            if mtype == "websocket.disconnect":
                break

            peer = room.peer_of(role)
            if peer is None:
                # Lone peer — drop the frame; receiver isn't there.
                continue

            try:
                if message.get("bytes") is not None:
                    await peer.send_bytes(message["bytes"])
                elif message.get("text") is not None:
                    await peer.send_text(message["text"])
            except Exception as exc:
                logger.warning("forward failed room=%s role=%s: %s", room_id, role, exc)
                # Don't tear down this side just because the peer hiccupped.
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("ws loop error room=%s role=%s: %s", room_id, role, exc)
    finally:
        async with room.lock:
            if room.slot(role) is ws:
                room.set_slot(role, None)
        await drop_room_if_empty(room_id)
        logger.info("leave room=%s role=%s", room_id, role)
