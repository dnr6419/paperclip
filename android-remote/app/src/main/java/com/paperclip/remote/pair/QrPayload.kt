package com.paperclip.remote.pair

import android.util.Base64

/**
 * Compact wire format for the first-pair QR.
 *
 *   v_byte(1=0x01) || room_id(6 ASCII) || id_pub(32) || eph_pub(32)  =  71 bytes
 *
 * Encoded as base64url (no padding), then prefixed with the URI scheme
 * `paperclip-remote://pair?d=...` so a future Android intent filter can
 * deep-link a scan straight into the pairing flow.
 *
 * 71 bytes fits comfortably in a QR-M @ ECC-Q ~125 alphanumeric chars.
 */
object QrPayload {
    private const val SCHEME_PREFIX = "paperclip-remote://pair?d="
    private const val VERSION: Byte = 0x01
    private const val ROOM_LEN = 6
    private const val KEY_LEN = 32
    const val TOTAL_LEN = 1 + ROOM_LEN + KEY_LEN + KEY_LEN

    data class Payload(val roomId: String, val idPub: ByteArray, val ephPub: ByteArray) {
        init {
            require(roomId.length == ROOM_LEN) { "roomId must be $ROOM_LEN chars" }
            require(idPub.size == KEY_LEN) { "idPub must be $KEY_LEN bytes" }
            require(ephPub.size == KEY_LEN) { "ephPub must be $KEY_LEN bytes" }
        }
        override fun equals(other: Any?): Boolean = this === other ||
            (other is Payload && roomId == other.roomId &&
             idPub.contentEquals(other.idPub) && ephPub.contentEquals(other.ephPub))
        override fun hashCode(): Int = (roomId.hashCode() * 31 +
            idPub.contentHashCode()) * 31 + ephPub.contentHashCode()
    }

    fun encode(p: Payload): String {
        val raw = ByteArray(TOTAL_LEN)
        raw[0] = VERSION
        for (i in 0 until ROOM_LEN) raw[1 + i] = p.roomId[i].code.toByte()
        System.arraycopy(p.idPub,  0, raw, 1 + ROOM_LEN,           KEY_LEN)
        System.arraycopy(p.ephPub, 0, raw, 1 + ROOM_LEN + KEY_LEN, KEY_LEN)
        val b64 = Base64.encodeToString(raw, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
        return SCHEME_PREFIX + b64
    }

    /** Tolerant decoder: accepts the full URI or just the base64url portion. */
    fun decode(input: String): Payload? {
        val b64 = input.removePrefix(SCHEME_PREFIX).trim()
        val raw = try {
            Base64.decode(b64, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
        } catch (_: IllegalArgumentException) {
            return null
        }
        if (raw.size != TOTAL_LEN || raw[0] != VERSION) return null
        val room = String(raw, 1, ROOM_LEN, Charsets.US_ASCII)
        if (RoomCode.normalize(room) != room) return null
        val idPub  = raw.copyOfRange(1 + ROOM_LEN,           1 + ROOM_LEN + KEY_LEN)
        val ephPub = raw.copyOfRange(1 + ROOM_LEN + KEY_LEN, TOTAL_LEN)
        return Payload(room, idPub, ephPub)
    }
}
