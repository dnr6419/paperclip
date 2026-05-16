package com.paperclip.remote.pair

import android.content.Context
import android.util.Base64
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import org.json.JSONArray
import org.json.JSONObject

/**
 * Encrypted on-disk record of peers this install has confirmed via the
 * safety-code step. Re-opening a saved peer skips the QR + fingerprint
 * confirmation and only re-runs the ephemeral exchange.
 *
 * Storage is `EncryptedSharedPreferences`, so the file at rest is
 * unreadable without the Android Keystore master key. We hold the whole
 * registry as one JSON blob — there will not be hundreds of peers and
 * the in-memory cost is negligible.
 */
class PeerRegistry(context: Context) {

    data class Peer(val alias: String, val idPub: ByteArray) {
        override fun equals(other: Any?): Boolean = this === other ||
            (other is Peer && alias == other.alias && idPub.contentEquals(other.idPub))
        override fun hashCode(): Int = alias.hashCode() * 31 + idPub.contentHashCode()
    }

    private val prefs = EncryptedSharedPreferences.create(
        context,
        PREFS_NAME,
        MasterKey.Builder(context).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
    )

    fun list(): List<Peer> {
        val raw = prefs.getString(KEY_PEERS, null) ?: return emptyList()
        return try {
            val arr = JSONArray(raw)
            buildList {
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    add(Peer(
                        alias = o.getString("alias"),
                        idPub = Base64.decode(o.getString("id_pub"), Base64.NO_WRAP),
                    ))
                }
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun findByIdPub(idPub: ByteArray): Peer? = list().firstOrNull { it.idPub.contentEquals(idPub) }

    fun save(peer: Peer) {
        val current = list().filterNot { it.idPub.contentEquals(peer.idPub) }
        val updated = current + peer
        writeAll(updated)
    }

    fun rename(idPub: ByteArray, newAlias: String) {
        val updated = list().map { if (it.idPub.contentEquals(idPub)) it.copy(alias = newAlias) else it }
        writeAll(updated)
    }

    fun delete(idPub: ByteArray) {
        writeAll(list().filterNot { it.idPub.contentEquals(idPub) })
    }

    private fun writeAll(peers: List<Peer>) {
        val arr = JSONArray()
        for (p in peers) {
            arr.put(JSONObject().apply {
                put("alias",  p.alias)
                put("id_pub", Base64.encodeToString(p.idPub, Base64.NO_WRAP))
            })
        }
        prefs.edit().putString(KEY_PEERS, arr.toString()).apply()
    }

    companion object {
        private const val PREFS_NAME = "paperclip-remote.peers"
        private const val KEY_PEERS  = "peers.json"
    }
}
