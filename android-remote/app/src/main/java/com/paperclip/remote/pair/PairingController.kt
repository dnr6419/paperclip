package com.paperclip.remote.pair

import com.paperclip.remote.crypto.Handshake
import com.paperclip.remote.crypto.IdentityStore
import com.paperclip.remote.session.SessionHolder
import com.paperclip.remote.transport.RelayClient
import com.paperclip.remote.transport.SecureChannel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import java.nio.charset.StandardCharsets

/**
 * Drives the v2 handshake from one side. Same code runs as controller
 * or controlled; the role is just a parameter that picks which slot
 * the relay places us in.
 *
 * Lifecycle:
 *
 *   start(role, relayUrl, roomId, peerIdPubExpected = …)
 *       │
 *       ├── opens RelayClient, generates ephemeral X25519, sends hello
 *       ├── awaits peer hello, parses HelloPayload
 *       ├── if peerIdPubExpected is provided (resumed pair) and matches,
 *       │   transitions to AwaitingConfirm with isResumed=true so the UI
 *       │   can skip the safety-code prompt
 *       ├── else exposes the 16-hex safety code for the user to compare
 *       │   on both phones, awaits confirm()
 *       └── on confirm(): derives session keys, saves peer to registry,
 *           installs SecureChannel into SessionHolder, state = Ready
 *
 *   Any failure transitions to Failed and tears down the socket.
 *
 * One PairingController instance handles one attempt. Tap "Pair again"
 * → spin up a fresh instance.
 */
class PairingController(
    private val identityStore: IdentityStore,
    private val peerRegistry: PeerRegistry,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
) {
    sealed interface State {
        data object Idle : State
        data object Connecting : State
        data object AwaitingPeerHello : State
        /** Show safety code; if [isResumed] true the UI may auto-confirm silently. */
        data class AwaitingConfirm(val safetyCode: String, val isResumed: Boolean) : State
        data class Ready(val peerIdPub: ByteArray, val isResumed: Boolean) : State
        data class Failed(val reason: String) : State
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    /**
     * Controlled side only: the QR payload the controller must scan.
     * Emitted once the local ephemeral key + socket are ready.
     */
    private val _qrPayload = MutableStateFlow<QrPayload.Payload?>(null)
    val qrPayload: StateFlow<QrPayload.Payload?> = _qrPayload.asStateFlow()

    private var relay: RelayClient? = null
    private var secureChannel: SecureChannel? = null
    private var pendingPeerHello: HelloPayload? = null
    private var pendingMyCanonical: ByteArray? = null
    private var pendingPeerCanonical: ByteArray? = null
    private var role: String? = null
    private var driver: Job? = null

    fun start(role: String, relayUrl: String, roomId: String, peerIdPubExpected: ByteArray? = null) {
        require(role == "controller" || role == "controlled") { "bad role" }
        cancel()  // discard any prior attempt

        this.role = role
        _state.value = State.Connecting

        driver = scope.launch {
            try {
                runHandshake(role, relayUrl, roomId, peerIdPubExpected)
            } catch (t: Throwable) {
                _state.value = State.Failed(t.message ?: t::class.simpleName ?: "unknown")
                tearDown()
            }
        }
    }

    /** Called from the UI after the user compares + accepts the safety code. */
    fun confirm() {
        val peerHello = pendingPeerHello ?: return
        val myCanonical = pendingMyCanonical ?: return
        val peerCanonical = pendingPeerCanonical ?: return
        val channel = secureChannel ?: return
        val r = role ?: return

        scope.launch {
            try {
                channel.completeHandshake(
                    myCanonicalHello = myCanonical,
                    peerCanonicalHello = peerCanonical,
                    peerIdPub = peerHello.idPub,
                    peerEphPub = peerHello.ephPub,
                )
                // Persist the peer (alias defaults to the safety code's first
                // group; the user can rename from the peer list later).
                val safety = Handshake.safetyCode(identityStore.keyPair.pub, peerHello.idPub)
                val existing = peerRegistry.findByIdPub(peerHello.idPub)
                if (existing == null) {
                    peerRegistry.save(PeerRegistry.Peer(
                        alias = "Peer-${safety.substring(0, 4)}",
                        idPub = peerHello.idPub,
                    ))
                }
                SessionHolder.set(channel)
                val isResumed = existing != null
                _state.value = State.Ready(peerHello.idPub, isResumed = isResumed)
            } catch (t: Throwable) {
                _state.value = State.Failed("handshake derive failed: ${t.message}")
                tearDown()
            }
        }
    }

    fun cancel() {
        driver?.cancel()
        driver = null
        tearDown()
    }

    private fun tearDown() {
        try { relay?.close(1000, "client_done") } catch (_: Exception) {}
        relay = null
        secureChannel = null
        pendingPeerHello = null
        pendingMyCanonical = null
        pendingPeerCanonical = null
        _qrPayload.value = null
    }

    private suspend fun runHandshake(
        role: String,
        relayUrl: String,
        roomId: String,
        peerIdPubExpected: ByteArray?,
    ) {
        val r = RelayClient()
        relay = r
        r.connect(relayUrl, roomId, role)
        // Wait until the socket is Open or fails fast.
        val openResult = withTimeoutOrNull(CONNECT_TIMEOUT_MS) {
            r.state.firstOrNull {
                it is RelayClient.State.Open || it is RelayClient.State.Failed || it is RelayClient.State.Closed
            }
        }
        when (openResult) {
            is RelayClient.State.Open -> {}
            is RelayClient.State.Failed -> { _state.value = State.Failed("connect failed: ${openResult.t.message}"); return }
            is RelayClient.State.Closed -> { _state.value = State.Failed("closed pre-hello (${openResult.code}: ${openResult.reason})"); return }
            null -> { _state.value = State.Failed("connect timeout"); return }
            else -> { _state.value = State.Failed("unexpected state $openResult"); return }
        }

        val myKp = identityStore.keyPair
        val channel = SecureChannel(
            relay = r,
            myRole = role,
            myIdPriv = myKp.priv,
            myIdPub = myKp.pub,
            scope = scope,
        )
        channel.start()
        secureChannel = channel

        // Controlled side: publish QR payload now that ephemeral is ready.
        if (role == "controlled") {
            _qrPayload.value = QrPayload.Payload(
                roomId = roomId,
                idPub  = myKp.pub,
                ephPub = channel.myEphPub,
            )
        }

        val myHello = HelloPayload(
            role = role,
            v = HelloPayload.PROTOCOL_VERSION,
            idPub = myKp.pub,
            ephPub = channel.myEphPub,
        )
        val myCanonical = myHello.toCanonicalJson()
        pendingMyCanonical = myCanonical
        r.sendRawText(String(myCanonical, StandardCharsets.UTF_8))
        _state.value = State.AwaitingPeerHello

        val peerText = withTimeoutOrNull(HELLO_TIMEOUT_MS) {
            channel.cleartextText.firstOrNull()
        } ?: run { _state.value = State.Failed("peer hello timeout"); return }

        val peerHello = HelloPayload.parse(peerText) ?: run {
            _state.value = State.Failed("peer hello parse failed"); return
        }
        if (peerHello.role == role) {
            _state.value = State.Failed("peer role mismatch (both = $role)"); return
        }
        pendingPeerHello = peerHello
        pendingPeerCanonical = peerText.toByteArray(StandardCharsets.UTF_8)

        if (peerIdPubExpected != null && !peerHello.idPub.contentEquals(peerIdPubExpected)) {
            _state.value = State.Failed("identity_mismatch — relay-side substitution?"); return
        }

        val isResumed = peerIdPubExpected != null ||
            peerRegistry.findByIdPub(peerHello.idPub) != null
        val safety = Handshake.safetyCode(myKp.pub, peerHello.idPub)
        _state.value = State.AwaitingConfirm(safety, isResumed)
    }

    companion object {
        private const val CONNECT_TIMEOUT_MS = 10_000L
        private const val HELLO_TIMEOUT_MS = 20_000L
    }
}
