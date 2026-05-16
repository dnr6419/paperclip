package com.paperclip.remote.files

import java.nio.charset.StandardCharsets

/**
 * Wire codec for the FILE_DATA (`0x02`) binary frame body. Per
 * `docs/PROTOCOL.md`:
 *
 *   id_len:u8 | id:utf8 | seq:u32 BE | data
 *
 * The transfer id is a short opaque string (we use 16 hex chars) that
 * lets concurrent transfers interleave on one connection. Sequence
 * numbers are per-transfer and start at 0.
 */
object FileChunk {
    const val MAX_CHUNK_BYTES = 16_384  // 16 KiB
    const val MAX_ID_BYTES = 255         // limited by u8 prefix

    fun encode(id: String, seq: Int, data: ByteArray): ByteArray {
        val idBytes = id.toByteArray(StandardCharsets.UTF_8)
        require(idBytes.size in 1..MAX_ID_BYTES) { "id length out of range" }
        require(seq >= 0) { "negative seq" }
        val out = ByteArray(1 + idBytes.size + 4 + data.size)
        out[0] = idBytes.size.toByte()
        System.arraycopy(idBytes, 0, out, 1, idBytes.size)
        val seqOff = 1 + idBytes.size
        out[seqOff]     = (seq ushr 24 and 0xff).toByte()
        out[seqOff + 1] = (seq ushr 16 and 0xff).toByte()
        out[seqOff + 2] = (seq ushr 8  and 0xff).toByte()
        out[seqOff + 3] = (seq         and 0xff).toByte()
        System.arraycopy(data, 0, out, seqOff + 4, data.size)
        return out
    }

    data class Decoded(val id: String, val seq: Int, val data: ByteArray) {
        override fun equals(other: Any?): Boolean = this === other ||
            (other is Decoded && id == other.id && seq == other.seq && data.contentEquals(other.data))
        override fun hashCode(): Int = (id.hashCode() * 31 + seq) * 31 + data.contentHashCode()
    }

    fun decode(payload: ByteArray): Decoded? {
        if (payload.size < 1 + 4) return null
        val idLen = payload[0].toInt() and 0xff
        if (idLen == 0 || idLen > MAX_ID_BYTES) return null
        if (payload.size < 1 + idLen + 4) return null
        val id = String(payload, 1, idLen, StandardCharsets.UTF_8)
        val seqOff = 1 + idLen
        val seq = ((payload[seqOff].toInt()     and 0xff) shl 24) or
                  ((payload[seqOff + 1].toInt() and 0xff) shl 16) or
                  ((payload[seqOff + 2].toInt() and 0xff) shl 8)  or
                  ((payload[seqOff + 3].toInt() and 0xff))
        if (seq < 0) return null
        val data = payload.copyOfRange(seqOff + 4, payload.size)
        return Decoded(id, seq, data)
    }
}
