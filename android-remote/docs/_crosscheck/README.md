# Crypto cross-check

Three regression gates for the wire protocol. Run them all to verify
that a code change did not drift from the spec in `../PROTOCOL.md`.

## 1. Python reference self-test

```bash
python3 ../protocol_reference.py            # 9 cases, prints "OK"
```

The reference implementation is the normative spec. If it disagrees
with PROTOCOL.md, fix PROTOCOL.md.

## 2. JVM ↔ Python cross-check

Validates that the JVM crypto primitives (which the Kotlin code on
Android compiles to) produce bit-identical outputs to the reference
on the same inputs.

```bash
javac CryptoCrossCheck.java
python3 run_crosscheck.py                   # 12 vectors, prints "ALL OK"
```

## 3. End-to-end through the real relay

Spins up `../../server/relay.py` on localhost and runs two clients
through it that execute the full handshake + AEAD-wrapped frame
exchange (TEXT + BINARY in both directions) plus the negative cases
(tamper, replay, wrong-direction key, MITM safety-code divergence).

```bash
pip install -r ../../server/requirements.txt websockets
python3 e2e_integration.py
```

A passing run prints `ALL OK -- full v2 protocol round-trip through
the real relay.` and exits 0.

---

Why a JVM cross-check rather than running the Kotlin code directly?
Building Android Kotlin needs an Android SDK + Gradle, which is heavy.
But Kotlin on Android invokes the exact same `javax.crypto.Cipher`
("ChaCha20-Poly1305") and `KeyAgreement` ("XDH") as plain Java does on
the JVM. Validating the math at the JCE layer transitively validates
the Kotlin code that wraps it.

Tink is used in the Kotlin code for X25519 + HKDF (the JCE has no HKDF
in the standard JDK). Tink and OpenJDK both implement RFC 7748 + RFC
5869, so their outputs match by construction; this harness re-checks
that assumption with concrete vectors.
