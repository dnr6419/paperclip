# Architecture

Why each module exists. Read this before changing structural code.

## Constraints driving the design

From `SEED.md`:

- **Same-owner trust** → no E2E media encryption, no account system, no
  per-message signing. WSS terminates trust at the relay.
- **≤ 500 ms input → screen target** → keep moving parts minimal. Every
  hop is latency. Server is a dumb forwarder; no transcoding, no buffering
  beyond OS sockets.
- **Self-hosted relay** → server must fit one Python file and a Dockerfile.
  No Redis, no DB, no auth provider.
- **Single APK with per-session role** → both Controller and Controlled
  code paths live in the same app, gated by which screen the user opens.

## Server (`android-remote/server`)

One process. One file: `relay.py`. Holds an in-memory `dict[str, Room]`.
Each `Room` has two slots (`controller`, `controlled`). Frames received
on one slot are forwarded to the other; the relay does not look inside.

Why FastAPI when there's no REST surface? Mostly habit, and `/healthz`
+ `uvicorn` integration come free. Could drop to bare `websockets` later
without changing the protocol.

Connection collisions (same role joining a room that already has that
role) **kick the old socket**. Mobile networks drop sockets often; the
newcomer is statistically more likely to be the real user. The old socket
gets close code `1001 replaced` so the client knows it wasn't a bug.

Rooms are deleted as soon as both slots go empty. There is no persistence
across server restarts — by design, room codes are one-time.

## Android app (`android-remote/app`)

Single APK. Three top-level user destinations:

- **Home** — pick role for this session.
- **Controller flow** — scan/enter code, then live view + input layer.
- **Controlled flow** — display code + QR, accept incoming session,
  start capture service.

```
app/src/main/java/com/paperclip/remote/
├── MainActivity.kt              navigation host
├── ui/
│   ├── HomeScreen.kt
│   ├── ControllerScreen.kt      QR scanner + video surface + input overlay
│   └── ControlledScreen.kt      room code display + status
├── transport/
│   ├── RelayClient.kt           WebSocket connect + send + receive loop
│   ├── Frame.kt                 sealed class for TEXT/BINARY frames
│   └── Protocol.kt              JSON message dataclasses + binary tags
├── capture/
│   ├── ScreenCaptureService.kt  foreground service, owns MediaProjection
│   └── H264Encoder.kt           MediaCodec wrapper, emits Annex-B NALs
├── playback/
│   └── H264Decoder.kt           controller-side MediaCodec → SurfaceTexture
├── input/
│   ├── RemoteInputService.kt    AccessibilityService: dispatchGesture etc.
│   └── InputMapper.kt           controller-side: scale taps to controlled px
├── files/
│   └── FileTransferManager.kt   chunk + reassemble + progress
└── pair/
    ├── RoomCode.kt              6-char base32 generator + validator
    └── QrCode.kt                ZXing thin wrapper
```

### Threading model

- WebSocket runs on a dedicated OkHttp dispatcher (single I/O thread).
- Encoder + decoder use their own `MediaCodec` callback threads.
- Input events from the controller UI are posted to the transport thread
  via a non-blocking `Channel<Frame>` (Kotlin coroutines).
- The AccessibilityService receives events on the main looper; it
  re-posts to a worker for `dispatchGesture` to avoid blocking.

Rule: **nothing on the main thread except UI state updates.** Compose
collects from `StateFlow`s fed by background coroutines.

### Why a foreground service for capture

`MediaProjection` requires it on API 31+. Also lets the user revoke
capture from the system notification, which is the right escape hatch.
Service stops itself when the WebSocket closes.

### Why a separate `AccessibilityService` for input

It is the only public API on stock Android that can synthesize taps
across app boundaries without root. The user enables it once in
Settings → Accessibility; the app deep-links them there. The service
sits idle until `RelayClient` hands it a `tap`/`swipe`/`key` frame.

## What the relay deliberately does NOT do

- No frame inspection, transcoding, or reformatting.
- No TURN-style media coalescing.
- No durable storage of files in transit.
- No metrics export beyond `/healthz`.

Each addition would either burn the latency budget or move trust into
the server, both of which conflict with the Seed.

## What the app deliberately does NOT do

- Persist room codes or peer identities across runs.
- Suppress or auto-accept the MediaProjection consent dialog.
- Inject keystrokes for password fields (Accessibility can, we won't —
  v1.1 IME passthrough will route through the proper IME framework).
- Auto-update over the air. Distribution is sideload-first.
