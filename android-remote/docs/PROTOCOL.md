# Wire protocol

Single WebSocket per peer to the relay. Relay forwards every frame
verbatim — it never inspects payloads. Both peers send + receive on the
same socket; multiplexing is done by frame type, not by separate channels.

## Endpoint

```
wss://<relay-host>/ws/<room_id>/<role>
```

- `room_id`: 6-character base32 code (no `0`, `1`, `I`, `O`, `L`, `U`).
- `role`: `controller` or `controlled`. Server rejects others.
- Connecting with a role already taken kicks the prior socket with close
  code `1001` reason `replaced`. (Lets a flapping mobile reconnect win.)

## Frame channels

WebSocket TEXT → JSON control messages.
WebSocket BINARY → first byte is a type tag, remainder is opaque to the
relay. Receiver dispatches by tag.

### TEXT — control (JSON, UTF-8)

Every message has a `t` field (type). Other fields depend on `t`. Unknown
`t` values are ignored by the receiver (forward-compatible).

#### `hello` — first message both peers must send

```json
{"t": "hello", "role": "controlled", "v": 1, "w": 1080, "h": 2400, "dpi": 480, "model": "Pixel 8"}
```

`v` is the protocol version. Receiving a higher `v` than supported: send
`bye` with reason `unsupported_version` and close. The controller side
includes `w`/`h`/`dpi` as best-known controller display geometry (for
absolute-coordinate mapping decisions).

#### `ready` — controlled side has consented and is about to start capture

```json
{"t": "ready"}
```

Sent after the user accepts the MediaProjection prompt. Until the
controller sees `ready`, it must not send input events.

#### `tap`, `swipe`, `key` — input from controller

```json
{"t": "tap",   "x": 540, "y": 1200, "ts": 1234567890}
{"t": "swipe", "x1": 540, "y1": 2000, "x2": 540, "y2": 600, "ms": 250, "ts": 1234567890}
{"t": "key",   "code": "BACK", "ts": 1234567890}
```

- Coordinates are in **controlled-side pixels**. Controller scales locally
  before sending using the `w`/`h` from the controlled side's `hello`.
- `key.code`: `BACK` | `HOME` | `RECENTS`. No keyboard codes in MVP (text
  input arrives via IME passthrough in v1.1).
- `ts`: controller-side `SystemClock.elapsedRealtime()` ms — used by the
  controlled side to drop stale events when its queue backs up.

#### `quality` — controller tunes the encoder

```json
{"t": "quality", "bitrate": 4000000, "fps": 30, "scale": 0.75}
```

Controlled side applies on next IDR. `scale` is fraction of native long
edge (e.g. `0.75` of 2400 → 1800).

#### `file_begin`, `file_end` — bracket a binary transfer

```json
{"t": "file_begin", "id": "u1", "name": "report.pdf", "size": 12345, "mime": "application/pdf"}
{"t": "file_end",   "id": "u1", "ok": true}
```

The `id` is a short ASCII tag chosen by the sender (≤ 16 bytes). Binary
chunks tagged with the same id follow. `file_end.ok=false` means abort —
receiver discards partial data.

#### `bye` — graceful close

```json
{"t": "bye", "reason": "user_disconnected"}
```

Known reasons: `user_disconnected`, `idle_timeout`, `unsupported_version`,
`permission_denied`, `internal_error`.

### BINARY — media + bulk

| byte 0 | name        | byte 1..n                                             |
|--------|-------------|-------------------------------------------------------|
| `0x01` | `VIDEO`     | H.264 Annex-B access unit (NALs concatenated)         |
| `0x02` | `FILE_DATA` | `id_len:u8 \| id:utf8 \| seq:u32 BE \| data`          |

#### `VIDEO` (0x01)

- Direction: `controlled → controller` only.
- One access unit per WebSocket message (don't fragment within a frame).
- IDR every ≤ 2 s and on every `quality` change.
- Receiver feeds the payload directly into `MediaCodec` decoder input.

#### `FILE_DATA` (0x02)

- Direction: either.
- `seq` starts at 0, increments by 1 per chunk, never gaps.
- Recommended chunk size: 64 KiB. The WebSocket layer already provides
  framing, so we don't add length prefixes for data.

## Connection lifecycle (happy path)

```
Controlled                Relay                 Controller
    │                       │                       │
    │── open ws/AB123/cd ──>│                       │
    │<──── (accepted) ──────│                       │
    │                       │<── open ws/AB123/ctrl │
    │                       │── (accepted) ────────>│
    │── hello {...} ───────>│── hello ────────────>│
    │<──────────────────── hello ─{...}─ from ctrl ─│
    │   [user accepts MediaProjection]              │
    │── ready ─────────────>│── ready ────────────>│
    │                       │                       │
    │<── tap {x,y} ─────────│<── tap {x,y} ────────│
    │── video 0x01 ────────>│── video 0x01 ───────>│
    │── video 0x01 ────────>│── video 0x01 ───────>│
    │   ...
    │── bye ───────────────>│── bye ──────────────>│
    │── close ─────────────>│                       │
    │                       │<───── close ──────────│
```

## Error rules

- Receiving binary before mutual `hello`+`ready`: drop.
- Receiving `tap`/`swipe`/`key` from a peer that announced `role=controlled`
  in its `hello`: drop (defense in depth — the relay already enforces this
  via the role slot, but never trust the relay).
- Backpressure: if the controlled side's outbound video buffer exceeds
  2 access units worth of pending bytes, drop the oldest non-IDR frame.
  Don't queue indefinitely — latency matters more than completeness.

## Versioning

Bump `hello.v` on **breaking** changes only. Forward-compatible additions
(new optional JSON fields, new `t` values, new binary type tags) keep `v`
the same. Receivers MUST ignore unknown fields and unknown `t` values.
