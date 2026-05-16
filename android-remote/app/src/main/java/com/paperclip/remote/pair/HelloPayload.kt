package com.paperclip.remote.pair

import android.util.Base64
import org.json.JSONObject
import java.nio.charset.StandardCharsets

/**
 * Cleartext `hello` exchanged before the AEAD layer is established.
 * See `docs/PROTOCOL.md §Handshake`.
 */
data class HelloPayload(
    val role: String,                  // "controller" | "controlled"
    val v: Int,                        // protocol version, must be 2
    val idPub: ByteArray,              // 32 bytes
    val ephPub: ByteArray,             // 32 bytes
    val w: Int? = null,
    val h: Int? = null,
    val dpi: Int? = null,
    val model: String? = null,
) {

    /**
     * Canonical JSON used both on the wire and for the transcript hash.
     * Sorted keys, no whitespace; optional fields are omitted when null
     * so two peers re-hashing each other's hellos see byte-identical
     * input regardless of which side originated each field set.
     */
    fun toCanonicalJson(): ByteArray {
        // org.json.JSONObject does NOT preserve insertion order on
        // toString — we hand-build a sorted serialization. The fields
        // we emit must match what the peer expects to hash.
        val fields = sortedMapOf<String, String>()
        fields["t"] = "\"hello\""
        fields["role"] = "\"$role\""
        fields["v"] = v.toString()
        fields["id_pub"] = "\"" + b64url(idPub) + "\""
        fields["eph_pub"] = "\"" + b64url(ephPub) + "\""
        if (w != null) fields["w"] = w.toString()
        if (h != null) fields["h"] = h.toString()
        if (dpi != null) fields["dpi"] = dpi.toString()
        if (model != null) fields["model"] = "\"" + escapeJson(model) + "\""
        val sb = StringBuilder("{")
        var first = true
        for ((k, v) in fields) {
            if (!first) sb.append(',')
            sb.append('"').append(k).append("\":").append(v)
            first = false
        }
        sb.append('}')
        return sb.toString().toByteArray(StandardCharsets.UTF_8)
    }

    override fun equals(other: Any?): Boolean = this === other ||
        (other is HelloPayload && role == other.role && v == other.v &&
         idPub.contentEquals(other.idPub) && ephPub.contentEquals(other.ephPub) &&
         w == other.w && h == other.h && dpi == other.dpi && model == other.model)

    override fun hashCode(): Int {
        var r = role.hashCode()
        r = r * 31 + v
        r = r * 31 + idPub.contentHashCode()
        r = r * 31 + ephPub.contentHashCode()
        return r
    }

    companion object {
        const val PROTOCOL_VERSION = 2

        fun parse(text: String): HelloPayload? {
            return try {
                val o = JSONObject(text)
                if (o.optString("t") != "hello") return null
                val v = o.getInt("v")
                if (v != PROTOCOL_VERSION) return null
                HelloPayload(
                    role  = o.getString("role"),
                    v     = v,
                    idPub  = Base64.decode(o.getString("id_pub"),  Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING),
                    ephPub = Base64.decode(o.getString("eph_pub"), Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING),
                    w     = if (o.has("w"))     o.getInt("w")     else null,
                    h     = if (o.has("h"))     o.getInt("h")     else null,
                    dpi   = if (o.has("dpi"))   o.getInt("dpi")   else null,
                    model = if (o.has("model")) o.getString("model") else null,
                )
            } catch (_: Exception) {
                null
            }
        }

        private fun b64url(b: ByteArray): String =
            Base64.encodeToString(b, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)

        private fun escapeJson(s: String): String =
            s.replace("\\", "\\\\").replace("\"", "\\\"")
    }
}
