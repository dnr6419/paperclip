package com.paperclip.remote.transport

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.encodeToString
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString

/**
 * Thin wrapper around OkHttp's WebSocket implementing the wire protocol.
 *
 * - JSON control messages flow through [outgoingControl] (in) and
 *   [incoming] (out — both TEXT and BINARY surface here, distinguished
 *   by which payload field is non-null).
 * - The class is role-agnostic — controller and controlled both use this
 *   client. Role is set at connect() time and burned into the URL.
 */
class RelayClient(
    private val httpClient: OkHttpClient = OkHttpClient(),
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
) {
    sealed interface State {
        data object Idle : State
        data object Connecting : State
        data object Open : State
        data class Closed(val code: Int, val reason: String) : State
        data class Failed(val t: Throwable) : State
    }

    sealed interface Incoming {
        data class Text(val message: ControlMessage) : Incoming
        data class Binary(val tag: Byte, val payload: ByteArray) : Incoming
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _incoming = MutableSharedFlow<Incoming>(
        replay = 0,
        extraBufferCapacity = 64,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val incoming: SharedFlow<Incoming> = _incoming.asSharedFlow()

    private var socket: WebSocket? = null

    fun connect(relayBaseUrl: String, roomId: String, role: String) {
        val url = "$relayBaseUrl/ws/$roomId/$role".replaceFirst("http", "ws")
        _state.value = State.Connecting
        socket = httpClient.newWebSocket(
            Request.Builder().url(url).build(),
            Listener(),
        )
    }

    fun send(msg: ControlMessage) {
        val socket = socket ?: return
        socket.send(ProtocolJson.encodeToString(msg))
    }

    fun sendBinary(tag: Byte, payload: ByteArray) {
        val socket = socket ?: return
        val framed = ByteArray(payload.size + 1).also {
            it[0] = tag
            System.arraycopy(payload, 0, it, 1, payload.size)
        }
        socket.send(framed.toByteString(0, framed.size))
    }

    fun close(code: Int = 1000, reason: String = "client_close") {
        socket?.close(code, reason)
        socket = null
    }

    private inner class Listener : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            _state.value = State.Open
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            val msg = runCatching {
                ProtocolJson.decodeFromString<ControlMessage>(text)
            }.getOrNull() ?: return
            scope.launch { _incoming.emit(Incoming.Text(msg)) }
        }

        override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
            if (bytes.size == 0) return
            val tag = bytes[0]
            val payload = bytes.substring(1).toByteArray()
            scope.launch { _incoming.emit(Incoming.Binary(tag, payload)) }
        }

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            webSocket.close(code, reason)
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            _state.value = State.Closed(code, reason)
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            _state.value = State.Failed(t)
        }
    }
}
