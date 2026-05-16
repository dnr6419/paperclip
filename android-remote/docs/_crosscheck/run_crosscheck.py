"""
Cross-check the JVM crypto implementation (CryptoCrossCheck.java) against
the Python reference (../protocol_reference.py).

Both must produce bit-identical outputs given identical inputs. If they
diverge, the Kotlin implementation that mirrors the JVM JCE will be
wrong on a real device too.

Run:
    cd android-remote/docs/_crosscheck
    javac CryptoCrossCheck.java
    python3 run_crosscheck.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # android-remote/
sys.path.insert(0, str(ROOT / "docs"))

import protocol_reference as ref  # noqa: E402


def canonical(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def main() -> int:
    # Fixed test vectors — both sides start from the same 32-byte scalars.
    a_id_priv  = bytes.fromhex("01" * 32)
    b_id_priv  = bytes.fromhex("02" * 32)
    a_eph_priv = bytes.fromhex("03" * 32)
    b_eph_priv = bytes.fromhex("04" * 32)

    a_id_pub_py  = ref.x25519_pub_from_priv(a_id_priv)
    b_id_pub_py  = ref.x25519_pub_from_priv(b_id_priv)
    a_eph_pub_py = ref.x25519_pub_from_priv(a_eph_priv)
    b_eph_pub_py = ref.x25519_pub_from_priv(b_eph_priv)

    hello_a = {"t": "hello", "role": "controller", "v": 2,
               "id_pub": a_id_pub_py.hex(), "eph_pub": a_eph_pub_py.hex()}
    hello_b = {"t": "hello", "role": "controlled", "v": 2,
               "id_pub": b_id_pub_py.hex(), "eph_pub": b_eph_pub_py.hex()}

    plaintext = b'{"t":"tap","x":540,"y":1200,"ts":12345}'

    # ---- Python expected values --------------------------------------------
    py_code = ref.safety_code(a_id_pub_py, b_id_pub_py)
    py_transcript = ref.transcript_hash(hello_a, hello_b)
    py_a_send, py_a_recv = ref.derive_session_keys(
        my_id_priv=a_id_priv, peer_id_pub=b_id_pub_py,
        my_eph_priv=a_eph_priv, peer_eph_pub=b_eph_pub_py,
        my_hello=hello_a, peer_hello=hello_b, my_role="controller",
    )
    py_b_send, py_b_recv = ref.derive_session_keys(
        my_id_priv=b_id_priv, peer_id_pub=a_id_pub_py,
        my_eph_priv=b_eph_priv, peer_eph_pub=a_eph_pub_py,
        my_hello=hello_b, peer_hello=hello_a, my_role="controlled",
    )
    py_wire_a = ref.wrap_frame(key=py_a_send, counter=0, kind=ref.KIND_TEXT,
                               tag_byte=None, plaintext=plaintext)
    py_wire_b = ref.wrap_frame(key=py_b_send, counter=42, kind=ref.KIND_BINARY,
                               tag_byte=0x01, plaintext=plaintext)

    # ---- Send vectors to Java, read back outputs ---------------------------
    stdin_payload = "\n".join([
        a_id_priv.hex(), b_id_priv.hex(),
        a_eph_priv.hex(), b_eph_priv.hex(),
        canonical(hello_a).hex(), canonical(hello_b).hex(),
        plaintext.hex(),
    ]) + "\n"
    proc = subprocess.run(
        ["java", "-cp", str(HERE), "CryptoCrossCheck"],
        input=stdin_payload, capture_output=True, text=True, check=True,
    )
    lines = proc.stdout.strip().splitlines()
    (j_a_id_pub, j_b_id_pub, j_code, j_transcript,
     j_a_send, j_a_recv, j_b_send, j_b_recv,
     j_wire_a, j_wire_b, j_dec_a, j_dec_b) = lines

    checks = [
        ("a_id_pub",   a_id_pub_py.hex(), j_a_id_pub),
        ("b_id_pub",   b_id_pub_py.hex(), j_b_id_pub),
        ("safety_code", py_code, j_code),
        ("transcript", py_transcript.hex(), j_transcript),
        ("a_send_key", py_a_send.hex(), j_a_send),
        ("a_recv_key", py_a_recv.hex(), j_a_recv),
        ("b_send_key", py_b_send.hex(), j_b_send),
        ("b_recv_key", py_b_recv.hex(), j_b_recv),
        ("wire_a_text", py_wire_a.hex(), j_wire_a),
        ("wire_b_binary", py_wire_b.hex(), j_wire_b),
        ("decrypt_a_text", plaintext.hex(), j_dec_a),
        ("decrypt_b_binary", plaintext.hex(), j_dec_b),
    ]
    fail = 0
    for name, expected, got in checks:
        if expected != got:
            fail += 1
            print(f"FAIL {name}\n  py:  {expected}\n  jvm: {got}")
        else:
            print(f"ok   {name}")
    if fail:
        print(f"\n{fail} mismatches", file=sys.stderr)
        return 1
    print("\nALL OK — Python reference and JVM crypto agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
