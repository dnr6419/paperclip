package com.paperclip.remote

import android.content.Context
import android.content.SharedPreferences

/**
 * Plain-prefs settings (not encrypted — these are UX choices, not
 * secrets). Only the relay URL lives here so far.
 */
class SettingsStore(context: Context) {
    private val prefs: SharedPreferences =
        context.getSharedPreferences("paperclip-remote.settings", Context.MODE_PRIVATE)

    var relayUrl: String
        get() = prefs.getString(KEY_RELAY_URL, "") ?: ""
        set(value) { prefs.edit().putString(KEY_RELAY_URL, value).apply() }

    companion object {
        private const val KEY_RELAY_URL = "relay_url"
    }
}
