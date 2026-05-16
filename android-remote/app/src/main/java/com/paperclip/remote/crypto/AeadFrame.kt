package com.paperclip.remote.crypto

import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * On-the-wire AEAD wrap from `docs/PROTOCOL.md §AEAD wrap`.
 *
 *   header   = u64_be(counter) || kind_byte [|| tag_byte if BINARY]
 *   nonce    = 4 zero bytes || u64_be(counter)
 *   wire     = header || ChaCha20-Poly1305(key, nonce, aad=header, plaintext)
 *
 * The receiver enforces strict monotonic counter (`> last_counter`).
 * ChaCha20-Poly1305 is provided by the platform on API 28+ via the
 * standard `javax.crypto.Cipher` API.
 */
object AeadFrame {
    const val KIND_TEXT: Byte = 0x54
    const val KIND_BINARY: Byte = 0x42

    private const val CIPHER = "ChaCha20-Poly1305"

    fun wrap(key: ByteArray, counter: Long, kind: Byte, tagByte: Byte?, plaintext: ByteArray): ByteArray {
        require(kind == KIND_TEXT || kind == KIND_BINARY) { "bad kind 0x%02x".format(kind) }
        require(counter >= 0L) { "counter must be unsigned-positive" }
        val header = headerBytes(counter, kind, tagByte)
        val cipher = Cipher.getInstance(CIPHER).apply {
            init(Cipher.ENCRYPT_MODE, SecretKeySpec(key, "ChaCha20"), IvParameterSpec(nonceFor(counter)))
            updateAAD(header)
        }
        val ct = cipher.doFinal(plaintext)
        return header + ct
    }

    /**
     * Result of a successful unwrap. `tagByte` is set only for BINARY frames.
     */
    data class Unwrapped(val counter: Long, val kind: Byte, val tagByte: Byte?, val plaintext: ByteArray)

    fun unwrap(key: ByteArray, lastCounter: Long, wire: ByteArray): Unwrapped {
        require(wire.size >= 9 + 16) { "frame too short" }
        val counter = readLongBe(wire, 0)
        if (counter <= lastCounter) error("replay: counter $counter <= last $lastCounter")
        val kind = wire[8]
        var offset = 9
        var tagByte: Byte? = null
        when (kind) {
            KIND_BINARY -> {
                require(wire.size >= 10 + 16) { "binary frame missing tag byte" }
                tagByte = wire[9]
                offset = 10
            }
            KIND_TEXT -> {}
            else -> error("unknown kind 0x%02x".format(kind))
        }
        val header = wire.copyOfRange(0, offset)
        val ct = wire.copyOfRange(offset, wire.size)
        val cipher = Cipher.getInstance(CIPHER).apply {
            init(Cipher.DECRYPT_MODE, SecretKeySpec(key, "ChaCha20"), IvParameterSpec(nonceFor(counter)))
            updateAAD(header)
        }
        val pt = cipher.doFinal(ct)
        return Unwrapped(counter, kind, tagByte, pt)
    }

    private fun headerBytes(counter: Long, kind: Byte, tagByte: Byte?): ByteArray {
        val out = ByteArray(if (kind == KIND_BINARY) 10 else 9)
        writeLongBe(out, 0, counter)
        out[8] = kind
        if (kind == KIND_BINARY) {
            require(tagByte != null) { "BINARY frame requires tagByte" }
            out[9] = tagByte
        }
        return out
    }

    private fun nonceFor(counter: Long): ByteArray {
        val n = ByteArray(12)
        writeLongBe(n, 4, counter)
        return n
    }

    private fun writeLongBe(buf: ByteArray, offset: Int, value: Long) {
        for (i in 0 until 8) {
            buf[offset + i] = ((value ushr ((7 - i) * 8)) and 0xff).toByte()
        }
    }

    private fun readLongBe(buf: ByteArray, offset: Int): Long {
        var v = 0L
        for (i in 0 until 8) {
            v = (v shl 8) or (buf[offset + i].toLong() and 0xff)
        }
        return v
    }
}
