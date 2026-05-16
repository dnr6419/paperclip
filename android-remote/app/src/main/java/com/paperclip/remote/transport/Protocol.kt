package com.paperclip.remote.transport

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonClassDiscriminator

/**
 * Control messages (TEXT frames). See docs/PROTOCOL.md.
 *
 * Unknown `t` values are ignored on receive — keep that path open by
 * decoding into [ControlMessage.Unknown] rather than throwing.
 */
val ProtocolJson = Json {
    encodeDefaults = false
    ignoreUnknownKeys = true
    classDiscriminator = "t"
}

@Serializable
sealed interface ControlMessage {
    // NOTE: `hello` is sent in cleartext *before* the AEAD layer is up,
    // so it lives in pair.HelloPayload rather than here. ControlMessage
    // covers only the post-handshake encrypted JSON frames.

    @Serializable @SerialName("ready")
    data object Ready : ControlMessage

    @Serializable @SerialName("tap")
    data class Tap(val x: Int, val y: Int, val ts: Long = 0L) : ControlMessage

    @Serializable @SerialName("swipe")
    data class Swipe(
        val x1: Int, val y1: Int, val x2: Int, val y2: Int,
        val ms: Int, val ts: Long = 0L,
    ) : ControlMessage

    @Serializable @SerialName("key")
    data class Key(val code: String, val ts: Long = 0L) : ControlMessage

    /**
     * Inject text at the currently focused EditText on the controlled
     * phone. Handled by the AccessibilityService via
     * `AccessibilityNodeInfo.ACTION_SET_TEXT`. Silently no-ops if no
     * focused text field exists.
     */
    @Serializable @SerialName("type")
    data class Type(val text: String, val ts: Long = 0L) : ControlMessage

    /**
     * Push the sender's clipboard onto the peer. Receiver attempts
     * `ClipboardManager.setPrimaryClip` — on Android Q+ this requires
     * the receiver app to be in the foreground (or an IME), so the
     * call may silently no-op on a background controlled phone. See
     * PROTOCOL.md §clipboard caveats.
     */
    @Serializable @SerialName("clipboard")
    data class Clipboard(val text: String, val ts: Long = 0L) : ControlMessage

    @Serializable @SerialName("quality")
    data class Quality(val bitrate: Int, val fps: Int, val scale: Double) : ControlMessage

    @Serializable @SerialName("file_begin")
    data class FileBegin(val id: String, val name: String, val size: Long, val mime: String) : ControlMessage

    @Serializable @SerialName("file_end")
    data class FileEnd(val id: String, val ok: Boolean) : ControlMessage

    @Serializable @SerialName("bye")
    data class Bye(val reason: String) : ControlMessage
}

/** Binary frame type tags (first byte of each WebSocket BINARY frame). */
object BinaryTag {
    const val VIDEO: Byte = 0x01
    const val FILE_DATA: Byte = 0x02
}
