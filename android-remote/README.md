# paperclip-remote

Android-to-Android remote control for your own devices. Self-hosted
WebSocket relay, one-time room codes, no account system.

Read `SEED.md` before touching this — it is the validated spec, and
code that disagrees with it is the bug.

## Current state

- [x] Wire protocol v2 (E2E, `docs/PROTOCOL.md`)
- [x] Relay server, docker-compose, round-trip verified
- [x] Android app scaffold — pairing, transport, UI shell
- [x] Identity store + handshake + AEAD wrap (E2E layer, cross-checked
      against Python reference)
- [x] MediaProjection capture + H.264 encoder + decoder pipeline
- [x] PairingController + PeerRegistry + safety-code UI
- [x] AccessibilityService input injection (taps, swipes, BACK/HOME/RECENTS)
- [ ] CameraX QR scanner on controller (typed-code path works today)
- [ ] Decoder Surface playback wired into ControllerScreen
- [ ] File transfer
- [ ] End-to-end on real hardware

See `SEED.md §11` for the definition of done.

## Run the relay

```bash
cd server
docker compose up -d
curl http://localhost:8765/healthz   # {"ok": true, "rooms": 0}
```

Production: terminate TLS at your reverse proxy and point both phones
at `wss://<your-host>/`.

## Quick local test without phones

```bash
cd server
pip install -r requirements.txt
uvicorn relay:app --host 127.0.0.1 --port 8765 &
python -c "
import asyncio, json, websockets
async def main():
    a = await websockets.connect('ws://127.0.0.1:8765/ws/ABCDEF/controller')
    b = await websockets.connect('ws://127.0.0.1:8765/ws/ABCDEF/controlled')
    await a.send(json.dumps({'t':'hello','role':'controller','v':1}))
    print('controlled received:', await b.recv())
asyncio.run(main())
"
```

## Build the Android app

Open `app/` in Android Studio Iguana or newer, let it generate the
gradle wrapper, run on a device with API 31+. There is no `gradlew`
checked in yet — adding one is one of the next commits.

## Layout

| Path                | Why                                           |
|---------------------|-----------------------------------------------|
| `SEED.md`           | Validated spec. Source of truth.              |
| `docs/PROTOCOL.md`  | Wire format. Server and client both rely on it.|
| `docs/ARCHITECTURE.md` | Module rationale.                          |
| `server/`           | FastAPI relay (single file).                  |
| `app/`              | Android (Kotlin + Compose).                   |
