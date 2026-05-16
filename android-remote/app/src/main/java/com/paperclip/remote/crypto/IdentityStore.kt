package com.paperclip.remote.crypto

import android.content.Context
import android.util.Base64
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * Long-lived X25519 identity keypair for this install. The private key
 * sits in EncryptedSharedPreferences backed by the Android Keystore,
 * which means it never leaves the device in plaintext form even on a
 * backup or a `run-as` debug pull.
 *
 * Generated on first read. Public key is also cached so the rest of the
 * app can fetch it without re-deriving on every call.
 */
class IdentityStore(context: Context) {

    private val prefs = EncryptedSharedPreferences.create(
        context,
        PREFS_NAME,
        MasterKey.Builder(context).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
    )

    val keyPair: Handshake.KeyPair by lazy { loadOrCreate() }

    private fun loadOrCreate(): Handshake.KeyPair {
        val storedPriv = prefs.getString(KEY_PRIV, null)
        if (storedPriv != null) {
            val priv = Base64.decode(storedPriv, Base64.NO_WRAP)
            return Handshake.KeyPair(priv = priv, pub = Handshake.publicFromPrivate(priv))
        }
        val fresh = Handshake.x25519KeyPair()
        prefs.edit()
            .putString(KEY_PRIV, Base64.encodeToString(fresh.priv, Base64.NO_WRAP))
            .putString(KEY_PUB,  Base64.encodeToString(fresh.pub,  Base64.NO_WRAP))
            .apply()
        return fresh
    }

    companion object {
        private const val PREFS_NAME = "paperclip-remote.identity"
        private const val KEY_PRIV = "x25519.priv"
        private const val KEY_PUB  = "x25519.pub"
    }
}
