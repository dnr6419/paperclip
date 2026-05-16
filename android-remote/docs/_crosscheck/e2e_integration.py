"""
End-to-end integration test for the paperclip-remote v2 protocol.

Spins up the real relay (server/relay.py) on a localhost port and runs
two clients through it that execute the full handshake + AEAD-wrapped
frame exchange, plus the negative cases (tamper, replay, wrong-
direction key, MITM safety-code divergence).

This is the strongest regression gate we have for the wire protocol —
if PROTOCOL.md changes in a way that breaks the bit-on-the-wire format,
the relay-level forwarding, or the cryptographic invariants, this test
fails loudly. Run from anywhere:

    python3 android-remote/docs/_crosscheck/e2e_integration.py

Exits 0 on success.
"""
import asyncio
import json
import pathlib
import sys
import threading
import time

import uvicorn
import websockets

HERE = pathlib.Path(__file__).resolve().parent
REMOTE = HERE.parent.parent  # android-remote/
sys.path.insert(0, str(REMOTE / "server"))
sys.path.insert(0, str(REMOTE / "docs"))

import relay  # noqa: E402
import protocol_reference as ref  # noqa: E402


def _start_relay(port: int) -> None:
    config = uvicorn.Config(relay.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True).start()
    time.sleep(0.6)


async def _run() -> int:
    PORT = 28765
    _start_relay(PORT)

    a_id_priv,  a_id_pub  = ref.x25519_keypair()
    b_id_priv,  b_id_pub  = ref.x25519_keypair()
    a_eph_priv, a_eph_pub = ref.x25519_keypair()
    b_eph_priv, b_eph_pub = ref.x25519_keypair()
    room = "ZK4VRT"  # 6-char alphanumeric, satisfies relay's room-id check

    # Both sides connect *before* either sends a hello, otherwise the
    # relay's drop-on-missing-peer rule eats the first message.
    a_ws = await websockets.connect(f"ws://127.0.0.1:{PORT}/ws/{room}/controller")
    b_ws = await websockets.connect(f"ws://127.0.0.1:{PORT}/ws/{room}/controlled")
    await asyncio.sleep(0.05)

    me_a = {"t": "hello", "role": "controller", "v": 2,
            "id_pub": a_id_pub.hex(), "eph_pub": a_eph_pub.hex()}
    me_b = {"t": "hello", "role": "controlled", "v": 2,
            "id_pub": b_id_pub.hex(), "eph_pub": b_eph_pub.hex()}
    await a_ws.send(json.dumps(me_a, sort_keys=True, separators=(",", ":")))
    await b_ws.send(json.dumps(me_b, sort_keys=True, separators=(",", ":")))

    peer_a = json.loads(await a_ws.recv())
    peer_b = json.loads(await b_ws.recv())
    assert peer_a["role"] == "controlled" and peer_b["role"] == "controller"
    assert bytes.fromhex(peer_a["id_pub"]) == b_id_pub
    assert bytes.fromhex(peer_b["id_pub"]) == a_id_pub
    print("[handshake] hellos exchanged through real relay")

    a_send, a_recv = ref.derive_session_keys(
        my_id_priv=a_id_priv, peer_id_pub=b_id_pub,
        my_eph_priv=a_eph_priv, peer_eph_pub=b_eph_pub,
        my_hello=me_a, peer_hello=peer_a, my_role="controller",
    )
    b_send, b_recv = ref.derive_session_keys(
        my_id_priv=b_id_priv, peer_id_pub=a_id_pub,
        my_eph_priv=b_eph_priv, peer_eph_pub=a_eph_pub,
        my_hello=me_b, peer_hello=peer_b, my_role="controlled",
    )
    assert a_send == b_recv and b_send == a_recv
    print(f"[handshake] session keys agree, safety code: {ref.safety_code(a_id_pub, b_id_pub)}")

    a_msgs = [
        (ref.KIND_TEXT,   None, b'{"t":"ready"}'),
        (ref.KIND_TEXT,   None, b'{"t":"tap","x":540,"y":1200,"ts":123}'),
        (ref.KIND_BINARY, 0x02, b"\xab" * 256),
    ]
    b_msgs = [
        (ref.KIND_TEXT,   None, b'{"t":"ready"}'),
        (ref.KIND_BINARY, 0x01, bytes(range(256)) * 8),  # 2 KiB fake NAL
    ]
    for ctr, (kind, tag, payload) in enumerate(a_msgs):
        await a_ws.send(ref.wrap_frame(key=a_send, counter=ctr,
                                       kind=kind, tag_byte=tag, plaintext=payload))
    for ctr, (kind, tag, payload) in enumerate(b_msgs):
        await b_ws.send(ref.wrap_frame(key=b_send, counter=ctr,
                                       kind=kind, tag_byte=tag, plaintext=payload))

    async def drain(ws, key, expected: int) -> list:
        out, last = [], -1
        for _ in range(expected):
            wire = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert not isinstance(wire, str), f"unexpected TEXT post-handshake: {wire[:80]}"
            ctr, kind, tag, pt = ref.unwrap_frame(key=key, last_counter=last, wire=wire)
            last = ctr
            out.append((kind, tag, pt))
        return out

    a_inbox = await drain(a_ws, a_recv, expected=len(b_msgs))
    b_inbox = await drain(b_ws, b_recv, expected=len(a_msgs))
    for i, (sent, recv) in enumerate(zip(b_msgs, a_inbox)):
        assert sent == recv, f"controller frame {i}: {sent[:2]} vs {recv[:2]}"
        print(f"  controller <- #{i}: kind=0x{recv[0]:02x} tag={recv[1]} {len(recv[2])}B OK")
    for i, (sent, recv) in enumerate(zip(a_msgs, b_inbox)):
        assert sent == recv, f"controlled frame {i}"
        print(f"  controlled <- #{i}: kind=0x{recv[0]:02x} tag={recv[1]} {len(recv[2])}B OK")

    await a_ws.close()
    await b_ws.close()

    print("\n--- negative cases ---")
    wire = ref.wrap_frame(key=a_send, counter=99, kind=ref.KIND_TEXT,
                          tag_byte=None, plaintext=b"hello")

    bad = bytearray(wire); bad[-1] ^= 1
    try:
        ref.unwrap_frame(key=b_recv, last_counter=-1, wire=bytes(bad))
    except Exception as e:
        print(f"  tamper rejected: {type(e).__name__}")
    else:
        print("  TAMPER NOT REJECTED"); return 1

    try:
        ref.unwrap_frame(key=b_recv, last_counter=99, wire=wire)
    except ValueError as e:
        assert "replay" in str(e)
        print(f"  replay rejected: {e}")
    else:
        print("  REPLAY NOT REJECTED"); return 1

    try:
        ref.unwrap_frame(key=a_recv, last_counter=-1, wire=wire)
    except Exception as e:
        print(f"  wrong-direction-key rejected: {type(e).__name__}")
    else:
        print("  WRONG KEY NOT REJECTED"); return 1

    _, fake_pub = ref.x25519_keypair()
    real_code = ref.safety_code(a_id_pub, b_id_pub)
    mitm_code = ref.safety_code(a_id_pub, fake_pub)
    assert real_code != mitm_code, "safety codes must diverge under MITM"
    print(f"  MITM safety code differs: real={real_code} vs mitm={mitm_code}")

    print("\nALL OK -- full v2 protocol round-trip through the real relay.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
