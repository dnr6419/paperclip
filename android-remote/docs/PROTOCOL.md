# Wire protocol (v2 — E2E)

Single WebSocket per peer to the relay. Relay forwards every frame
verbatim — it never inspects payloads. Both peers send + receive on the
same socket; multiplexing is done by frame type, not by separate channels.

**Confidentiality model.** The relay sees only ciphertext for everything
sent after the cryptographic handshake completes. Plaintext layers
described below are what each peer encrypts/decrypts locally; on the
wire they are wrapped (see §AEAD wrap).

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

#### `hello` — first message both peers must send (CLEARTEXT)

```json
{
  "t": "hello",
  "role": "controlled",
  "v": 2,
  "id_pub": "BASE64URL(32-byte X25519 identity pubkey)",
  "eph_pub": "BASE64URL(32-byte X25519 ephemeral pubkey)",
  "w": 1080, "h": 2400, "dpi": 480, "model": "Pixel 8"
}
```

`v` is the protocol version. Receiving a higher `v` than supported: send
`bye` with reason `unsupported_version` and close. The controller side
includes `w`/`h`/`dpi` as best-known controller display geometry.

`hello` is **the only message sent in cleartext.** It carries the
material needed to derive a session key (see §Handshake). Everything
after the local `ready` is AEAD-wrapped.

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

## Handshake (v2)

Identity keys are long-lived per install, stored encrypted on each
device. Ephemeral keys are fresh per session.

```
KDF_input =
  "paperclip-remote v2 session-key"     (literal)
  || transcript_hash                     (32 bytes, SHA-256 over both hellos
                                          in canonical lexicographic-key JSON,
                                          shorter side first)
  || X25519(my_id_priv,  peer_id_pub)    (32 bytes — static)
  || X25519(my_eph_priv, peer_eph_pub)   (32 bytes — ephemeral)

session_key = HKDF-SHA256(salt=transcript_hash, ikm=KDF_input, info=role, L=32)
```

A different `info` per direction (`"controller"` vs `"controlled"`)
yields two distinct keys, one per direction. Each direction has its own
nonce counter starting at 0.

**First pairing (no prior knowledge):**

1. Controlled side displays a QR with
   `room_id || id_pub_controlled || nonce_qr`. The controller side scans.
2. Both phones display the same **16-hex safety code** (64 bits,
   rendered `XXXX-XXXX-XXXX-XXXX`) derived from the two identity
   pubkeys:

   ```
   safety_code(a_pub, b_pub) =
     SHA-256("paperclip-remote v2 safety" || lo || hi).hex()[:16]
   where lo, hi = sorted(a_pub, b_pub)  (byte-lex)
   ```

   The user must confirm both phones show the same code; if a relay
   substituted either key, the codes will differ. 64 bits gives
   ≈ 1.8×10¹⁹ work for an offline second-preimage, comfortable margin
   for the QR-display window.
3. After acceptance, store the peer's `id_pub` keyed by alias.

**Resumed pairing (identity pubkeys already known):**

1. Both sides skip the safety-phrase step and silently verify that the
   peer's `id_pub` in the cleartext `hello` matches the stored value.
2. Mismatch → close with `bye.reason = "identity_mismatch"`. Do not
   automatically re-pair; require the user to explicitly delete the
   stored peer and rescan.

## AEAD wrap

Every WebSocket message sent after the local `ready` is wrapped:

```
on-the-wire frame =
  u64_be(nonce_counter)               (8 bytes)
  || frame_kind_byte                  (1 byte: 0x54 TEXT, 0x42 BINARY+tag)
  || (if BINARY) original tag byte    (1 byte)
  || ChaCha20-Poly1305(
        key   = session_key_for_my_direction,
        nonce = 4 zero bytes || u64_be(nonce_counter)    (12 bytes total),
        aad   = u64_be(nonce_counter) || frame_kind_byte [|| tag],
        plaintext = original payload
     )                                (M + 16 bytes)
```

The whole frame is sent as a BINARY WebSocket message regardless of
whether the inner payload was TEXT or BINARY. The receiver:

1. Reads the 8-byte counter, rejects if not `> last_seen_counter`
   (replay/reorder defense — strict monotonic).
2. Reads kind byte; if 0x42, reads the original tag.
3. AEAD-decrypts with peer's direction key; on failure, close with
   `bye.reason = "decrypt_failure"`.
4. Dispatches the plaintext exactly as the cleartext spec above
   describes.

Nonces never wrap. A session that approaches 2⁶³ frames sent (≈ never
in practice) must close with `bye.reason = "rekey_needed"`. v2 does not
specify rekey; v3 may.

## Versioning

Bump `hello.v` on **breaking** changes only. Forward-compatible additions
(new optional JSON fields, new `t` values, new binary type tags) keep `v`
the same. Receivers MUST ignore unknown fields and unknown `t` values.

v1 (cleartext) is gone. Mixed-version connections close with
`unsupported_version`.
