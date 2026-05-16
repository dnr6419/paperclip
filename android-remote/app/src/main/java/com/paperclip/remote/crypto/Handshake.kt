package com.paperclip.remote.crypto

import com.google.crypto.tink.subtle.Hkdf
import com.google.crypto.tink.subtle.X25519
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/**
 * Pure functions implementing the v2 handshake described in
 * `docs/PROTOCOL.md`. Mirror of `docs/protocol_reference.py` — if these
 * two ever diverge, the Python file wins (it's the normative spec).
 */
object Handshake {

    fun x25519KeyPair(): KeyPair {
        val priv = X25519.generatePrivateKey()
        val pub = X25519.publicFromPrivate(priv)
        return KeyPair(priv = priv, pub = pub)
    }

    fun publicFromPrivate(priv: ByteArray): ByteArray = X25519.publicFromPrivate(priv)

    fun dh(myPriv: ByteArray, peerPub: ByteArray): ByteArray =
        X25519.computeSharedSecret(myPriv, peerPub)

    /**
     * SHA-256 over `canonical(my_hello) || canonical(peer_hello)`, with
     * the shorter byte string concatenated first (length-then-lex). The
     * caller must produce canonical JSON (sorted keys, no whitespace).
     */
    fun transcriptHash(myCanonicalHello: ByteArray, peerCanonicalHello: ByteArray): ByteArray {
        val a = myCanonicalHello
        val b = peerCanonicalHello
        val (lo, hi) = if (lengthThenLexLessOrEqual(a, b)) a to b else b to a
        val md = MessageDigest.getInstance("SHA-256")
        md.update(lo)
        md.update(hi)
        return md.digest()
    }

    /**
     * Returns (sendKey, recvKey). `myRole` is "controller" or "controlled".
     * The HKDF info string is the role of the side that *encrypts* with
     * the key, so `sendKey` uses `myRole` and `recvKey` uses the peer's.
     */
    fun deriveSessionKeys(
        myIdPriv: ByteArray, peerIdPub: ByteArray,
        myEphPriv: ByteArray, peerEphPub: ByteArray,
        transcript: ByteArray,
        myRole: String,
    ): Pair<ByteArray, ByteArray> {
        require(myRole == "controller" || myRole == "controlled") { "invalid role: $myRole" }
        val staticDh = dh(myIdPriv, peerIdPub)
        val ephDh = dh(myEphPriv, peerEphPub)
        val ikm = ByteArray(KDF_PREFIX.size + transcript.size + staticDh.size + ephDh.size).apply {
            var o = 0
            System.arraycopy(KDF_PREFIX, 0, this, o, KDF_PREFIX.size); o += KDF_PREFIX.size
            System.arraycopy(transcript, 0, this, o, transcript.size); o += transcript.size
            System.arraycopy(staticDh, 0, this, o, staticDh.size); o += staticDh.size
            System.arraycopy(ephDh, 0, this, o, ephDh.size)
        }
        val peerRole = if (myRole == "controller") "controlled" else "controller"
        val sendKey = Hkdf.computeHkdf("HMACSHA256", ikm, transcript, myRole.toByteArray(StandardCharsets.UTF_8), 32)
        val recvKey = Hkdf.computeHkdf("HMACSHA256", ikm, transcript, peerRole.toByteArray(StandardCharsets.UTF_8), 32)
        return sendKey to recvKey
    }

    /**
     * 16-hex-character (64-bit) safety code, formatted XXXX-XXXX-XXXX-XXXX,
     * symmetric in the two pubkey arguments. See PROTOCOL.md §Handshake.
     */
    fun safetyCode(aPub: ByteArray, bPub: ByteArray): String {
        val (lo, hi) = if (byteLexLessOrEqual(aPub, bPub)) aPub to bPub else bPub to aPub
        val md = MessageDigest.getInstance("SHA-256").apply {
            update(SAFETY_PREFIX)
            update(lo)
            update(hi)
        }
        val hex = md.digest().sliceArray(0..7).joinToString("") { "%02X".format(it) }
        return "${hex.substring(0, 4)}-${hex.substring(4, 8)}-${hex.substring(8, 12)}-${hex.substring(12, 16)}"
    }

    data class KeyPair(val priv: ByteArray, val pub: ByteArray) {
        override fun equals(other: Any?): Boolean = this === other ||
            (other is KeyPair && priv.contentEquals(other.priv) && pub.contentEquals(other.pub))
        override fun hashCode(): Int = priv.contentHashCode() * 31 + pub.contentHashCode()
    }

    private val KDF_PREFIX = "paperclip-remote v2 session-key".toByteArray(StandardCharsets.UTF_8)
    private val SAFETY_PREFIX = "paperclip-remote v2 safety".toByteArray(StandardCharsets.UTF_8)

    private fun lengthThenLexLessOrEqual(a: ByteArray, b: ByteArray): Boolean {
        if (a.size != b.size) return a.size < b.size
        return byteLexLessOrEqual(a, b)
    }

    private fun byteLexLessOrEqual(a: ByteArray, b: ByteArray): Boolean {
        val n = minOf(a.size, b.size)
        for (i in 0 until n) {
            val ai = a[i].toInt() and 0xff
            val bi = b[i].toInt() and 0xff
            if (ai != bi) return ai < bi
        }
        return a.size <= b.size
    }
}
