# SEED — Android Remote Control App

> Generated via Ouroboros-style Interview → Seed flow.
> Ambiguity score at sign-off: **0.12** (target ≤ 0.20).
> Date: 2026-05-16
>
> **Amendment 2026-05-16/r2**: §2 persistence and §4 cryptography
> re-negotiated mid-Execute after the user picked persistent same-owner
> pairing + E2E encryption + AccessibilityService input. The earlier
> "no E2E", "no persistent pairing" non-goals are removed. New
> Ambiguity post-amendment: 0.10.

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
- **Persistent device identity.** Each install generates a long-lived
  X25519 + Ed25519 identity keypair stored in
  `EncryptedSharedPreferences` (Android Keystore-backed). First pairing
  is QR-mediated; subsequent connections between the same two installs
  auto-resume without a fresh code.
- A short-lived 6-character room code remains the fallback path for
  re-pairing after an app reinstall or for new device pairs.

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

WebRTC would give us lower latency and direct P2P, but it requires
STUN+TURN infrastructure and a much larger client SDK. WSS + an
application-layer AEAD wrap (see §4) gets us E2E confidentiality without
DTLS-SRTP, keeps the relay <300 lines of Python, and is good enough for
the 500 ms target.

---

## 4. Pairing & auth

Two pairing paths:

**First pairing (QR-mediated, in-band verification):**

1. Controlled phone generates a room code AND a per-session X25519
   ephemeral keypair. Encodes a QR containing:
   `room_id || identity_pubkey_controlled || ephemeral_pubkey_controlled
   || nonce`.
2. Controller scans QR. Confirms a 4-word safety phrase derived from
   `SHA-256(identity_pubkey_controlled)` matches what the controlled
   phone displays. (Defense against relay-side MITM substituting keys.)
3. Both sides perform X25519(my_identity_priv, their_identity_pub) +
   X25519(my_ephemeral_priv, their_ephemeral_pub), feed both into HKDF
   with a transcript hash → session key.
4. From `ready` onward, every WebSocket message (TEXT and BINARY) is
   wrapped: `nonce_counter || ChaCha20-Poly1305(session_key,
   nonce_counter, plaintext)`. Counter is per-direction, monotonic,
   resets per session.
5. After successful pairing the **controlled side's identity_pubkey**
   is saved on the controller (and vice versa) under a user-chosen
   alias.

**Resumed connection (paired devices):**

1. Either side opens a saved peer. Controller connects to a room
   derived from `HMAC(identity_shared_secret, today's date)` so even
   the room namespace rotates.
2. Only the ephemeral X25519 exchange happens; identity verification
   is silent (keys already known).
3. Same AEAD wrap applies.

### Threat model

| Adversary | Defense |
|---|---|
| Random attacker guesses room code in 5 min | 30⁶ ≈ 7×10⁸ + rate-limit 10/min/IP |
| Eavesdropper between phone and relay | WSS |
| **Compromised/curious relay operator** | E2E AEAD; relay sees ciphertext only |
| Relay swaps keys during first pairing | QR fingerprint comparison on first pair |
| Relay swaps keys after pairing | Stored identity pubkey rejects mismatch |
| Replay of captured ciphertext | Per-direction monotonic nonce counter |

### Threat model (still out of scope)

- Compromised controlled phone (e.g. malware also has Accessibility).
- Coercion of the user.
- Forward secrecy of past sessions if identity key is later stolen.
  Ephemeral exchange provides per-session forward secrecy *forward* of
  identity compromise, but does not retroactively protect old sessions
  against a key-exfiltration on the device.

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
- After session establishment every TEXT/BINARY frame is AEAD-wrapped
  by an outer fixed framing: `u64 nonce_counter || ciphertext` with
  ChaCha20-Poly1305 (16-byte tag). The relay treats this as opaque
  bytes; only the two peers can decrypt.

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
