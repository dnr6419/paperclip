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
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import java.nio.charset.StandardCharsets

/**
 * Drives the v2 handshake from one side. Same code runs as controller
 * or controlled; the role is just a parameter that picks which slot
 * the relay places us in.
 *
 * Two confirmation paths:
 *
 *   - **First pair** — no prior identity. UI shows the 16-hex safety
 *     code, user compares both phones and calls [confirm].
 *   - **Resumed pair** — peer identity already known (from QR
 *     fingerprint or PeerRegistry). Handshake auto-confirms silently;
 *     UI never enters AwaitingConfirm.
 *
 * Auto-resume: once Ready, the controller subscribes to the underlying
 * RelayClient state and retries with exponential backoff if the socket
 * closes unexpectedly. Retries use the same room id + the now-known
 * peer identity, so the user doesn't have to do anything during a Wi-Fi
 * flap.
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
        /** Show safety code; if [isResumed] true the controller auto-confirms internally. */
        data class AwaitingConfirm(val safetyCode: String, val isResumed: Boolean) : State
        data class Ready(
            val peerIdPub: ByteArray,
            val isResumed: Boolean,
            val peerWidthPx: Int,
            val peerHeightPx: Int,
        ) : State
        /** Auto-resume in progress; controller will retry shortly. */
        data class Reconnecting(val attempt: Int, val reason: String) : State
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
    private var watchdog: Job? = null

    /** Saved so auto-resume can re-run [runHandshake] without UI input. */
    private data class StartParams(
        val role: String, val relayUrl: String, val roomId: String,
        val peerIdPubExpected: ByteArray?, val myWidthPx: Int?, val myHeightPx: Int?,
    )
    private var lastStart: StartParams? = null

    fun start(
        role: String,
        relayUrl: String,
        roomId: String,
        peerIdPubExpected: ByteArray? = null,
        myWidthPx: Int? = null,
        myHeightPx: Int? = null,
    ) {
        require(role == "controller" || role == "controlled") { "bad role" }
        cancel()  // discard any prior attempt

        this.role = role
        lastStart = StartParams(role, relayUrl, roomId, peerIdPubExpected, myWidthPx, myHeightPx)
        _state.value = State.Connecting

        driver = scope.launch {
            try {
                runHandshake(role, relayUrl, roomId, peerIdPubExpected, myWidthPx, myHeightPx)
            } catch (t: Throwable) {
                _state.value = State.Failed(t.message ?: t::class.simpleName ?: "unknown")
                tearDown()
            }
        }
    }

    /** Called from the UI after the user compares + accepts the safety code. */
    fun confirm() {
        scope.launch { finalizeHandshake() }
    }

    private suspend fun finalizeHandshake() {
        val peerHello = pendingPeerHello ?: return
        val myCanonical = pendingMyCanonical ?: return
        val peerCanonical = pendingPeerCanonical ?: return
        val channel = secureChannel ?: return

        try {
            channel.completeHandshake(
                myCanonicalHello = myCanonical,
                peerCanonicalHello = peerCanonical,
                peerIdPub = peerHello.idPub,
                peerEphPub = peerHello.ephPub,
            )
            val safety = Handshake.safetyCode(identityStore.keyPair.pub, peerHello.idPub)
            val existing = peerRegistry.findByIdPub(peerHello.idPub)
            if (existing == null) {
                peerRegistry.save(PeerRegistry.Peer(
                    alias = "Peer-${safety.substring(0, 4)}",
                    idPub = peerHello.idPub,
                ))
            }
            SessionHolder.set(channel)
            _state.value = State.Ready(
                peerIdPub = peerHello.idPub,
                isResumed = existing != null,
                peerWidthPx  = peerHello.w ?: 1080,
                peerHeightPx = peerHello.h ?: 2400,
            )
            startWatchdog()
        } catch (t: Throwable) {
            _state.value = State.Failed("handshake derive failed: ${t.message}")
            tearDown()
        }
    }

    fun cancel() {
        watchdog?.cancel(); watchdog = null
        driver?.cancel(); driver = null
        lastStart = null  // user-initiated cancel disables auto-resume
        tearDown()
        if (_state.value !is State.Idle) _state.value = State.Idle
    }

    private fun tearDown() {
        try { relay?.close(1000, "client_done") } catch (_: Exception) {}
        relay = null
        secureChannel = null
        pendingPeerHello = null
        pendingMyCanonical = null
        pendingPeerCanonical = null
        _qrPayload.value = null
        SessionHolder.clear()
    }

    private suspend fun runHandshake(
        role: String,
        relayUrl: String,
        roomId: String,
        peerIdPubExpected: ByteArray?,
        myWidthPx: Int?,
        myHeightPx: Int?,
    ) {
        val r = RelayClient()
        relay = r
        r.connect(relayUrl, roomId, role)
        val openResult = withTimeoutOrNull(CONNECT_TIMEOUT_MS) {
            r.state.firstOrNull {
                it is RelayClient.State.Open || it is RelayClient.State.Failed || it is RelayClient.State.Closed
            }
        }
        when (openResult) {
            is RelayClient.State.Open -> {}
            is RelayClient.State.Failed  -> { _state.value = State.Failed("connect failed: ${openResult.t.message}"); return }
            is RelayClient.State.Closed  -> { _state.value = State.Failed("closed pre-hello (${openResult.code}: ${openResult.reason})"); return }
            null -> { _state.value = State.Failed("connect timeout"); return }
            else -> { _state.value = State.Failed("unexpected state $openResult"); return }
        }

        val myKp = identityStore.keyPair
        val channel = SecureChannel(
            relay = r, myRole = role,
            myIdPriv = myKp.priv, myIdPub = myKp.pub,
            scope = scope,
        )
        channel.start()
        secureChannel = channel

        if (role == "controlled") {
            _qrPayload.value = QrPayload.Payload(roomId = roomId,
                idPub = myKp.pub, ephPub = channel.myEphPub)
        }

        val myHello = HelloPayload(
            role = role, v = HelloPayload.PROTOCOL_VERSION,
            idPub = myKp.pub, ephPub = channel.myEphPub,
            w = myWidthPx, h = myHeightPx,
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

        val knownPeer = peerRegistry.findByIdPub(peerHello.idPub) != null
        val isResumed = peerIdPubExpected != null || knownPeer
        val safety = Handshake.safetyCode(myKp.pub, peerHello.idPub)
        _state.value = State.AwaitingConfirm(safety, isResumed)

        // Resumed pair → auto-confirm. First pair → wait for user.
        if (isResumed) finalizeHandshake()
    }

    /**
     * Watches the relay socket. If it closes while we were Ready and
     * the user didn't cancel, kicks off [autoResume] on the
     * supervisor scope.
     */
    private fun startWatchdog() {
        watchdog?.cancel()
        val r = relay ?: return
        watchdog = scope.launch {
            r.state.collect { s ->
                if (s is RelayClient.State.Closed || s is RelayClient.State.Failed) {
                    if (_state.value is State.Ready && lastStart != null) {
                        autoResume()
                    }
                    return@collect
                }
            }
        }
    }

    private suspend fun autoResume() {
        val params = lastStart ?: return
        val knownPeer = params.peerIdPubExpected
            ?: pendingPeerHello?.idPub
            ?: run { _state.value = State.Failed("auto-resume: no peer id"); return }

        SessionHolder.clear()
        try { relay?.close(1000, "auto_resume") } catch (_: Exception) {}

        val backoffMs = longArrayOf(2_000L, 4_000L, 8_000L, 16_000L, 30_000L)
        var attempt = 0
        while (attempt < MAX_RESUME_ATTEMPTS) {
            attempt += 1
            val wait = backoffMs.getOrElse(attempt - 1) { backoffMs.last() }
            _state.value = State.Reconnecting(attempt, reason = "socket dropped")
            delay(wait)
            try {
                runHandshake(
                    role = params.role,
                    relayUrl = params.relayUrl,
                    roomId = params.roomId,
                    peerIdPubExpected = knownPeer,
                    myWidthPx = params.myWidthPx,
                    myHeightPx = params.myHeightPx,
                )
                // runHandshake transitions to Ready (via auto-confirm) on success.
                if (_state.value is State.Ready) return
                // Otherwise it set Failed; treat as a retry.
            } catch (_: Throwable) {
                // continue
            }
        }
        _state.value = State.Failed("auto-resume gave up after $attempt attempts")
    }

    companion object {
        private const val CONNECT_TIMEOUT_MS = 10_000L
        private const val HELLO_TIMEOUT_MS = 20_000L
        private const val MAX_RESUME_ATTEMPTS = 12     // ~5 minutes with backoff
    }
}
