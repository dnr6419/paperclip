# Tracking — paperclip-remote work units

Repo convention is `feat: <description> (DOF-NNN)` per recent
`backtesting/` commits (e.g. `feat: add Coppock Curve and CMF
strategies (DOF-297)`). The Android remote-control work was developed
under the working prefix `DOF-R-NNN` since Linear isn't reachable from
the development container; map to the real numbers when filing.

## Work units shipped on `feature/android-remote-control`

| Working ID | Commit(s) | Scope |
|---|---|---|
| DOF-R-1 | `5c0fd60` | Seed: validated spec + Ouroboros-style Interview answers (Ambiguity 0.12) |
| DOF-R-2 | `36ae588`, `bb03f34` | Wire protocol + architecture + README v0 |
| DOF-R-3 | `71fac00` | FastAPI WebSocket relay + docker-compose |
| DOF-R-4 | `09b7040` | Android app scaffold (Compose, Nav, Manifest, theme) |
| DOF-R-5 | `6719756` | Seed amendment r2 — persistent pairing + E2E AEAD |
| DOF-R-6 | `effa02f` | E2E crypto: X25519 handshake + ChaCha20-Poly1305 wrap (12-vector cross-check) |
| DOF-R-7 | `67aa653` | MediaProjection capture + H.264 encode + decode |
| DOF-R-8 | `4ab8751` | E2E integration test preserved as regression gate |
| DOF-R-9 | `ced04b4` | PairingController + PeerRegistry + AccessibilityService input |
| DOF-R-10 | `8d6b11c` | InputMapper + decoder Surface playback + tap/swipe send (13 JVM cases) |
| DOF-R-11 | `dfa3f82` | CameraX QR scanner |
| DOF-R-12 | `0046c62` | Bidirectional file transfer (12 JVM cases) |
| DOF-R-13 | `7bea284` | README rewrite with Mermaid diagrams + usage guide |
| DOF-R-14 | `1dcc536` | Text injection + clipboard push (v1.1 SEED items) |

## Open follow-ups (not on this branch yet)

| Working ID | Scope |
|---|---|
| DOF-R-15 | On-device acceptance run per SEED §11 DoD on at least one hardware pair, recorded under `docs/device-acceptance.md` |
| DOF-R-16 | Hardware-key chip row (BACK/HOME/RECENTS) on ControllerScreen |
| DOF-R-17 | Quality control (`ControlMessage.Quality` UI surface to tune bitrate/fps live) |
| DOF-R-18 | Gradle wrapper checked in so CI can build without Android Studio |
| DOF-R-19 | Foreground reconnect/auto-resume on `feature/android-remote-control` socket flap |

## Commit convention going forward

```
<type>(remote): <subject>      ← repo style; "remote" scope distinguishes
                                  this subtree from the backtesting work

<body explaining why, with verification notes where applicable>

(DOF-NNN)                       ← real Linear ticket if known
```

Types in use:
- `feat` — user-visible behaviour change
- `fix` — bug fix
- `docs` — README / SEED / PROTOCOL / inline docs
- `test` — regression gates, no behaviour change
- `seed` — Seed amendment (rare; rewrites the validated spec)
- `chore` — build, deps, infra
