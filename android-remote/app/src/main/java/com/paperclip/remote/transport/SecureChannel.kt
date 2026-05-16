package com.paperclip.remote.transport

import com.paperclip.remote.crypto.AeadFrame
import com.paperclip.remote.crypto.Handshake
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicLong

/**
 * Wraps a [RelayClient] with the v2 AEAD layer. Use sequence:
 *
 *   1. construct, [start] passing the bare RelayClient
 *   2. [sendHello] with the local hello (cleartext)
 *   3. [onPeerHello] once the cleartext hello from the peer arrives
 *   4. derive keys via [completeHandshake] — channel is now "live"
 *   5. [sendText] / [sendBinary] go through AEAD wrap; [incoming]
 *      yields decrypted frames
 *
 * All counter and key state is held here so callers don't have to.
 */
class SecureChannel(
    private val relay: RelayClient,
    private val myRole: String,            // "controller" | "controlled"
    private val myIdPriv: ByteArray,
    private val myIdPub: ByteArray,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
) {
    sealed interface Plain {
        data class Text(val message: ControlMessage) : Plain
        data class Binary(val tag: Byte, val payload: ByteArray) : Plain
    }

    private val ephemeral = Handshake.x25519KeyPair()
    val myEphPub: ByteArray get() = ephemeral.pub

    private var sendKey: ByteArray? = null
    private var recvKey: ByteArray? = null
    private val sendCounter = AtomicLong(0L)
    @Volatile private var lastRecvCounter: Long = -1L

    private val _incoming = MutableSharedFlow<Plain>(
        replay = 0, extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val incoming: SharedFlow<Plain> = _incoming.asSharedFlow()

    /**
     * Raw cleartext TEXT frames from the relay (used to read the peer's
     * initial `hello`). Stops being relevant after [completeHandshake];
     * any post-handshake TEXT received here is a protocol violation by
     * the peer.
     */
    val cleartextText: SharedFlow<String>
        get() = _cleartextText.asSharedFlow()
    private val _cleartextText = MutableSharedFlow<String>(
        replay = 1, extraBufferCapacity = 4, onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

    fun start() {
        scope.launch {
            relay.incoming.collect { frame ->
                when (frame) {
                    is RelayClient.Incoming.Text -> _cleartextText.emit(frame.text)
                    is RelayClient.Incoming.Binary -> handleBinary(frame.bytes)
                }
            }
        }
    }

    private suspend fun handleBinary(wire: ByteArray) {
        val keys = recvKey ?: return  // pre-handshake noise; drop
        val unwrapped = try {
            AeadFrame.unwrap(keys, lastRecvCounter, wire)
        } catch (_: Exception) {
            relay.close(1008, "decrypt_failure")
            return
        }
        lastRecvCounter = unwrapped.counter
        val plain = when (unwrapped.kind) {
            AeadFrame.KIND_TEXT -> {
                val text = String(unwrapped.plaintext, StandardCharsets.UTF_8)
                val msg = runCatching {
                    ProtocolJson.decodeFromString<ControlMessage>(text)
                }.getOrNull() ?: return
                Plain.Text(msg)
            }
            AeadFrame.KIND_BINARY -> Plain.Binary(unwrapped.tagByte!!, unwrapped.plaintext)
            else -> return
        }
        _incoming.emit(plain)
    }

    /**
     * Build the canonical cleartext `hello` object containing both identity
     * and ephemeral public keys. Caller serializes + sends as a normal
     * TEXT frame via [sendCleartextJson].
     */
    fun buildHello(extras: JsonObject = JsonObject(emptyMap())): JsonObject = buildJsonObject {
        put("t", "hello")
        put("role", myRole)
        put("v", 2)
        put("id_pub", b64(myIdPub))
        put("eph_pub", b64(myEphPub))
        for ((k, v) in extras) put(k, v)
    }

    fun sendCleartextJson(obj: JsonObject) {
        relay.sendRawText(Json.encodeToString(JsonObject.serializer(), obj))
    }

    /**
     * Called once with the peer's parsed hello to derive session keys.
     * After this returns, subsequent [sendText]/[sendBinary] calls are
     * AEAD-wrapped.
     */
    fun completeHandshake(myCanonicalHello: ByteArray, peerCanonicalHello: ByteArray, peerIdPub: ByteArray, peerEphPub: ByteArray) {
        val transcript = Handshake.transcriptHash(myCanonicalHello, peerCanonicalHello)
        val (send, recv) = Handshake.deriveSessionKeys(
            myIdPriv = myIdPriv,
            peerIdPub = peerIdPub,
            myEphPriv = ephemeral.priv,
            peerEphPub = peerEphPub,
            transcript = transcript,
            myRole = myRole,
        )
        sendKey = send
        recvKey = recv
    }

    fun sendText(message: ControlMessage) {
        val keys = sendKey ?: error("handshake not complete")
        val plaintext = ProtocolJson.encodeToString(message).toByteArray(StandardCharsets.UTF_8)
        val ctr = sendCounter.getAndIncrement()
        val wire = AeadFrame.wrap(keys, ctr, AeadFrame.KIND_TEXT, null, plaintext)
        relay.sendRawBinary(wire)
    }

    fun sendBinary(tag: Byte, payload: ByteArray) {
        val keys = sendKey ?: error("handshake not complete")
        val ctr = sendCounter.getAndIncrement()
        val wire = AeadFrame.wrap(keys, ctr, AeadFrame.KIND_BINARY, tag, payload)
        relay.sendRawBinary(wire)
    }

    private fun b64(b: ByteArray): String = android.util.Base64.encodeToString(b, android.util.Base64.NO_WRAP or android.util.Base64.NO_PADDING or android.util.Base64.URL_SAFE)
}
