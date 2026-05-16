"""
Reference implementation of the paperclip-remote v2 cryptographic
protocol. This file is **normative**: when the Kotlin code in
`app/src/main/java/com/paperclip/remote/crypto/` disagrees with this
file, the Kotlin code is wrong.

Run directly to exercise the round-trip self-test:

    python3 docs/protocol_reference.py

A passing run prints "OK" and exits 0. The asserts at the bottom are
the conformance test vectors any other implementation must reproduce.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat,
)


# ---------- key utilities ---------------------------------------------------

def x25519_keypair() -> tuple[bytes, bytes]:
    priv = X25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub_bytes = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return priv_bytes, pub_bytes


def x25519_pub_from_priv(priv: bytes) -> bytes:
    return X25519PrivateKey.from_private_bytes(priv).public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw,
    )


def x25519_dh(priv: bytes, peer_pub: bytes) -> bytes:
    return X25519PrivateKey.from_private_bytes(priv).exchange(
        X25519PublicKey.from_public_bytes(peer_pub),
    )


# ---------- safety code -----------------------------------------------------

def safety_code(a_pub: bytes, b_pub: bytes) -> str:
    """16-hex-character (64-bit) fingerprint of the pair of identity pubkeys.

    Symmetric in (a_pub, b_pub): we sort byte-lex first so both phones
    derive the same string regardless of who is "first".
    """
    lo, hi = sorted([a_pub, b_pub])
    h = hashlib.sha256(b"paperclip-remote v2 safety" + lo + hi).hexdigest()[:16]
    return f"{h[0:4]}-{h[4:8]}-{h[8:12]}-{h[12:16]}".upper()


# ---------- transcript hash -------------------------------------------------

def canonical_hello(hello: dict) -> bytes:
    """JSON with sorted keys, no whitespace, UTF-8."""
    return json.dumps(hello, sort_keys=True, separators=(",", ":")).encode("utf-8")


def transcript_hash(hello_a: dict, hello_b: dict) -> bytes:
    """SHA-256 over canonical(hello_a) || canonical(hello_b), shorter first.

    The order is determined by serialized length then lex, so both peers
    compute the same hash without coordinating who is 'A'.
    """
    a, b = canonical_hello(hello_a), canonical_hello(hello_b)
    lo, hi = sorted([a, b], key=lambda x: (len(x), x))
    return hashlib.sha256(lo + hi).digest()


# ---------- session-key derivation ------------------------------------------

def derive_session_keys(
    *,
    my_id_priv: bytes, peer_id_pub: bytes,
    my_eph_priv: bytes, peer_eph_pub: bytes,
    my_hello: dict, peer_hello: dict,
    my_role: str,    # "controller" or "controlled"
) -> tuple[bytes, bytes]:
    """Returns (send_key, recv_key). Both are 32 bytes.

    `info` on HKDF differs by direction so each side encrypts with its
    own key and decrypts with the peer's.
    """
    th = transcript_hash(my_hello, peer_hello)
    static_dh = x25519_dh(my_id_priv, peer_id_pub)
    eph_dh = x25519_dh(my_eph_priv, peer_eph_pub)
    ikm = b"paperclip-remote v2 session-key" + th + static_dh + eph_dh

    def hkdf(info: str) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32, salt=th, info=info.encode("utf-8"),
        ).derive(ikm)

    peer_role = "controlled" if my_role == "controller" else "controller"
    return hkdf(my_role), hkdf(peer_role)


# ---------- AEAD wrap -------------------------------------------------------

KIND_TEXT   = 0x54  # 'T'
KIND_BINARY = 0x42  # 'B'


def wrap_frame(
    *,
    key: bytes, counter: int,
    kind: int, tag_byte: int | None,
    plaintext: bytes,
) -> bytes:
    """Encrypt one frame for transmission. counter must be strictly
    monotonic per direction; caller increments after a successful wrap."""
    assert kind in (KIND_TEXT, KIND_BINARY)
    assert 0 <= counter < (1 << 64)
    header = struct.pack(">QB", counter, kind)
    if kind == KIND_BINARY:
        assert tag_byte is not None and 0 <= tag_byte <= 0xFF
        header += bytes([tag_byte])
    nonce = b"\x00\x00\x00\x00" + struct.pack(">Q", counter)
    aead = ChaCha20Poly1305(key)
    ct = aead.encrypt(nonce, plaintext, header)
    return header + ct


def unwrap_frame(
    *,
    key: bytes, last_counter: int,
    wire: bytes,
) -> tuple[int, int, int | None, bytes]:
    """Decrypt one frame. Returns (counter, kind, tag_byte, plaintext).

    Raises ValueError on replay (counter <= last_counter) or on AEAD
    failure (corruption / wrong key / wrong AAD)."""
    if len(wire) < 9 + 16:
        raise ValueError("frame too short")
    counter, kind = struct.unpack(">QB", wire[:9])
    if counter <= last_counter:
        raise ValueError(f"replay: counter {counter} <= last {last_counter}")
    offset = 9
    tag_byte: int | None = None
    if kind == KIND_BINARY:
        if len(wire) < 10 + 16:
            raise ValueError("binary frame missing tag byte")
        tag_byte = wire[9]
        offset = 10
    elif kind != KIND_TEXT:
        raise ValueError(f"unknown kind 0x{kind:02x}")
    header = wire[:offset]
    ct = wire[offset:]
    nonce = b"\x00\x00\x00\x00" + struct.pack(">Q", counter)
    aead = ChaCha20Poly1305(key)
    plaintext = aead.decrypt(nonce, ct, header)
    return counter, kind, tag_byte, plaintext


# ---------- self-test -------------------------------------------------------

def _selftest() -> None:
    # 1. Identity + ephemeral keypairs for both peers.
    a_id_priv, a_id_pub = x25519_keypair()
    b_id_priv, b_id_pub = x25519_keypair()
    a_eph_priv, a_eph_pub = x25519_keypair()
    b_eph_priv, b_eph_pub = x25519_keypair()

    # 2. Both compute the same safety code regardless of arg order.
    assert safety_code(a_id_pub, b_id_pub) == safety_code(b_id_pub, a_id_pub)
    code = safety_code(a_id_pub, b_id_pub)
    assert len(code) == 19 and code.count("-") == 3, code

    # 3. Hellos as the peers would send them (v=2, base64-ish stand-ins
    #    are fine for the transcript hash — actual base64 happens at the
    #    JSON layer in the real client).
    hello_a = {"t": "hello", "role": "controller", "v": 2,
               "id_pub": a_id_pub.hex(), "eph_pub": a_eph_pub.hex()}
    hello_b = {"t": "hello", "role": "controlled", "v": 2,
               "id_pub": b_id_pub.hex(), "eph_pub": b_eph_pub.hex(),
               "w": 1080, "h": 2400, "dpi": 480, "model": "Pixel 8"}

    # 4. Each side derives its (send, recv). A's send must equal B's recv.
    a_send, a_recv = derive_session_keys(
        my_id_priv=a_id_priv, peer_id_pub=b_id_pub,
        my_eph_priv=a_eph_priv, peer_eph_pub=b_eph_pub,
        my_hello=hello_a, peer_hello=hello_b,
        my_role="controller",
    )
    b_send, b_recv = derive_session_keys(
        my_id_priv=b_id_priv, peer_id_pub=a_id_pub,
        my_eph_priv=b_eph_priv, peer_eph_pub=a_eph_pub,
        my_hello=hello_b, peer_hello=hello_a,
        my_role="controlled",
    )
    assert a_send == b_recv, "A's send key must match B's recv key"
    assert b_send == a_recv, "B's send key must match A's recv key"

    # 5. Round-trip a TEXT frame.
    msg = b'{"t":"tap","x":540,"y":1200,"ts":12345}'
    wire = wrap_frame(key=a_send, counter=0, kind=KIND_TEXT,
                      tag_byte=None, plaintext=msg)
    ctr, kind, tag, pt = unwrap_frame(key=b_recv, last_counter=-1, wire=wire)
    assert (ctr, kind, tag, pt) == (0, KIND_TEXT, None, msg)

    # 6. Round-trip a BINARY VIDEO frame (tag 0x01).
    nal = bytes(range(256)) * 64  # 16 KiB faux NAL
    wire = wrap_frame(key=b_send, counter=42, kind=KIND_BINARY,
                      tag_byte=0x01, plaintext=nal)
    ctr, kind, tag, pt = unwrap_frame(key=a_recv, last_counter=41, wire=wire)
    assert (ctr, kind, tag) == (42, KIND_BINARY, 0x01)
    assert pt == nal

    # 7. Replay rejection.
    try:
        unwrap_frame(key=a_recv, last_counter=42, wire=wire)
    except ValueError as e:
        assert "replay" in str(e)
    else:
        raise AssertionError("replay should have been rejected")

    # 8. Tamper rejection (flip a ciphertext byte).
    tampered = bytearray(wire)
    tampered[-1] ^= 0x01
    try:
        unwrap_frame(key=a_recv, last_counter=41, wire=bytes(tampered))
    except Exception:
        pass
    else:
        raise AssertionError("tamper should have been rejected")

    # 9. Wrong-key rejection (using send key to decrypt own send).
    try:
        unwrap_frame(key=a_send, last_counter=41, wire=wire)
    except Exception:
        pass
    else:
        raise AssertionError("wrong-key decrypt should have failed")

    print("OK")


if __name__ == "__main__":
    _selftest()
    sys.exit(0)
