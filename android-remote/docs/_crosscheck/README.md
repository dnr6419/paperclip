# Crypto cross-check

Validates that the JVM crypto primitives (which Kotlin will compile to
on a real device) produce bit-identical results to the normative Python
reference in `../protocol_reference.py`.

```bash
javac CryptoCrossCheck.java
python3 run_crosscheck.py
```

A passing run prints `ALL OK` and exits 0. If you change anything in
`docs/PROTOCOL.md` that affects the wire format, update **both** sides
here and re-run; the harness will tell you which output drifted.

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
