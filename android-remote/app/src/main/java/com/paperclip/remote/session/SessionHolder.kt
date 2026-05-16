package com.paperclip.remote.session

import com.paperclip.remote.transport.BinaryTag
import com.paperclip.remote.transport.SecureChannel
import java.util.concurrent.atomic.AtomicReference

/**
 * Process-wide handle to the currently active session. ScreenCaptureService
 * and the UI both reach in to read or update it without an explicit DI
 * graph — the app is small enough that a single-slot atomic is plainer
 * than dragging in Hilt for one thing.
 *
 * Lifetime: set when the controlled-side user accepts a session, cleared
 * when the WebSocket closes. There is only ever one active session.
 */
object SessionHolder {
    private val current = AtomicReference<SecureChannel?>(null)

    fun set(channel: SecureChannel) { current.set(channel) }
    fun clear() { current.set(null) }
    fun get(): SecureChannel? = current.get()

    /**
     * Send a binary frame on the active session, or drop it silently if
     * there is no session. Used by the encoder callback — we never want
     * an encoder pump to crash because the session went away.
     */
    fun sendBinary(tag: Byte, payload: ByteArray) {
        current.get()?.sendBinary(tag, payload)
    }

    fun sendVideo(payload: ByteArray) = sendBinary(BinaryTag.VIDEO, payload)
}
