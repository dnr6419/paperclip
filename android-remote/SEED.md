# SEED — Android Remote Control App

> Generated via Ouroboros-style Interview → Seed flow.
> Ambiguity score at sign-off: **0.12** (target ≤ 0.20).
> Date: 2026-05-16

This document is the **validated specification**. Code lives downstream of
it; conflicts between code and this file are bugs in the code until this
file is amended.

---

## 1. Purpose

A single Android APK that lets **one user** control their own Android devices
(phone ↔ tablet, primary phone ↔ secondary phone) from each other over the
internet via a self-hosted relay. Same-owner trust model — not designed for
helping someone else, not designed for IT helpdesk, not designed for low-trust
peers.

### Non-goals

- Cross-platform clients (no iOS, desktop, browser viewer). Android-only.
- Multi-tenant SaaS. Each user runs their own relay.
- Helping non-technical users on the controlled side. Both devices are owned
  by the same person who will grant the required permissions.
- Sub-200ms gaming-grade latency. Sub-1s file transfers of multi-GB files.
- Concealing the screen-mirror permission prompt (the system-level
  MediaProjection dialog will appear on the controlled phone every time the
  OS demands it; the app will not pursue tricks to suppress it).

---

## 2. Roles & symmetry

- Single APK installed on both phones.
- Per-session role choice: either side can be **Controller** (sees + drives)
  or **Controlled** (shares screen + accepts input).
- No persistent device pairing in v1. Each session is established fresh.

---

## 3. Transport & topology

```
[Controller phone] ──WSS──> [Relay] <──WSS── [Controlled phone]
```

- Relay: **self-hosted**, Python (FastAPI + uvicorn), single process,
  stateless beyond in-memory room registry. Docker-compose deployable
  alongside the existing paperclip stack.
- Protocol: **WebSocket Secure**. Relay forwards frames verbatim, never
  inspects payload. No media transcoding on the server.
- One **room** = one session. Room is destroyed when both peers disconnect.
- TLS termination is the relay operator's responsibility (reverse proxy or
  uvicorn `--ssl-*` flags). The app refuses non-`wss://` URLs in production
  builds.

### Why not WebRTC

WebRTC would give us lower latency and direct P2P, but it requires STUN+TURN
infrastructure and the same-owner trust model doesn't benefit from
end-to-end media encryption (the user already trusts their own server).
WSS keeps the server <300 lines of Python and is good enough for the
500 ms target.

---

## 4. Pairing & auth

- Controlled phone generates a **one-time 6-character room code** (base32,
  no ambiguous chars) and renders it as text + QR.
- Controller scans QR or types code → connects to same room.
- Controlled phone shows a system confirmation ("Accept connection from
  this controller?") before accepting any input or starting capture.
- Room code is invalidated immediately after both peers join, OR after 5
  minutes if unused, OR after either peer disconnects.
- No user accounts, no persistence of pairings. Re-pair next session.

### Threat model (in scope)

- Random attacker guessing a 6-char room code in the 5-minute window:
  32⁶ ≈ 10⁹ → with rate limiting (10 tries/min per IP), infeasible.
- Eavesdropper between phone and relay: prevented by WSS.
- Operator of the relay (= the user themselves): trusted by definition.

### Threat model (explicitly out of scope)

- Compromised controlled phone (e.g. malware also has Accessibility) —
  this app cannot defend against that.
- Coercion of the user to share their room code.

---

## 5. Capabilities (MVP scope)

| Feature           | Direction              | Status |
|-------------------|------------------------|--------|
| Screen mirror     | Controlled → Controller | MVP    |
| Tap / swipe       | Controller → Controlled | MVP    |
| Hardware keys (BACK / HOME / RECENTS) | Controller → Controlled | MVP    |
| File transfer     | Bidirectional          | MVP    |
| Text input (IME passthrough) | Controller → Controlled | v1.1   |
| Clipboard sync    | Bidirectional          | v1.1   |
| Audio mirror      | Controlled → Controller | future |
| Multi-device fanout | 1 controller → N controlled | future |

---

## 6. Performance targets (acceptance criteria)

End-to-end on a residential 100 Mbit symmetric link, relay co-located in
the same country, both phones on 5 GHz Wi-Fi:

| Metric                                 | Target           | Stretch        |
|----------------------------------------|------------------|----------------|
| Input → on-screen feedback (median)    | ≤ 500 ms         | ≤ 300 ms       |
| Video frame rate                       | ≥ 20 fps         | ≥ 30 fps       |
| Video resolution (long edge)           | ≥ 720 px         | source-native  |
| Video bitrate ceiling                  | 4 Mbps           | adaptive       |
| File transfer (1 GiB)                  | completes < 5 min| < 3 min        |
| Cold-start to first frame              | ≤ 8 s            | ≤ 5 s          |
| Pairing time (code → connected)        | ≤ 15 s           | ≤ 10 s         |

A build is "MVP done" iff all **Target** column rows pass on at least one
hardware pair we own.

---

## 7. Platform constraints

- **minSdk 31** (Android 12), targetSdk current.
- **Kotlin** + Jetpack Compose UI.
- Required runtime permissions:
  - `FOREGROUND_SERVICE` + `FOREGROUND_SERVICE_MEDIA_PROJECTION`
  - `POST_NOTIFICATIONS` (foreground service notification)
  - `CAMERA` (QR scan on controller side only)
  - **Accessibility Service** (input injection on controlled side only) —
    granted via Settings → Accessibility, app surfaces a deep-link.
  - MediaProjection consent (per-session OS prompt, cannot be persisted).
- No root, no Shizuku.
- Single APK, all code in `android-remote/app`.
- Server lives in `android-remote/server` and is independent of the rest of
  the paperclip repo (no shared Python modules with `backtesting/`).

---

## 8. Wire protocol (summary)

Full spec lives in `docs/PROTOCOL.md`. Two frame channels over the same
WebSocket:

- **TEXT frames** carry JSON control messages: `hello`, `ready`, `tap`,
  `swipe`, `key`, `file_begin`, `file_end`, `quality`, `bye`.
- **BINARY frames** carry a 1-byte type tag + payload: `0x01` H.264 NAL
  access units, `0x02` file data chunks tagged by transfer id.

The protocol is intentionally minimal — adding a new message type should
not require a server release.

---

## 9. Repository layout

```
android-remote/
├── SEED.md                  this file
├── README.md                quickstart
├── docs/
│   ├── PROTOCOL.md          wire protocol
│   └── ARCHITECTURE.md      module-by-module rationale
├── server/                  Python relay
│   ├── relay.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
└── app/                     Android (Gradle KTS)
    ├── build.gradle.kts
    ├── settings.gradle.kts
    └── src/main/...
```

This subtree is intentionally self-contained — it has no Python imports
from `backtesting/` or `strategies/`, and the dashboard does not surface
remote-control state. Treat it as a peer project that happens to share a
git repo.

---

## 10. Out-of-scope decisions deferred to Execute phase

These do **not** affect Ambiguity and will be resolved during the Design
sub-step of Execute (Double Diamond):

- Exact H.264 encoder parameters (profile, GOP, B-frames).
- Whether to use `Choreographer` or raw `SurfaceTexture` callback for
  the controller-side decode loop.
- Whether `OkHttp.WebSocket` or `ktor-client-websockets` ships better
  binary backpressure on Android 12+.
- Room code character set (base32 vs. word-list à la Magic Wormhole).
- Whether file chunks should be size-prefixed or count on WebSocket
  message boundaries (probably the latter — WSS already frames).

---

## 11. Definition of done — MVP

The branch `feature/android-remote-control` merges to `main` when:

1. All §6 **Target** metrics pass on the developer's hardware.
2. `server/` has a `docker-compose up` path that just works.
3. README documents the 5-step quickstart end-to-end.
4. A 90-second screen recording demonstrates pairing → control → file send.
5. No `TODO` or `FIXME` blocks remain in shipped code.

---

## 12. Evaluation gates (Ouroboros)

Before each commit on this branch, ask:

- **Mechanical**: does it build, lint, and run?
- **Semantic**: does it advance one of §5's MVP rows or §11's DoD bullets?
  If neither, why is it here?
- **Consensus**: would a reviewer agree the change is the smallest one
  that moves the metric? (Avoid premature abstraction; this is a
  300-line server, not a framework.)
