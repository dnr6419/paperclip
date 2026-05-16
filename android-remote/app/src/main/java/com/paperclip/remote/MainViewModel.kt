package com.paperclip.remote

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.paperclip.remote.crypto.IdentityStore
import com.paperclip.remote.files.FileTransferManager
import com.paperclip.remote.pair.PairingController
import com.paperclip.remote.pair.PeerRegistry
import com.paperclip.remote.pair.RoomCode
import com.paperclip.remote.session.SessionHolder
import com.paperclip.remote.transport.ControlMessage
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch

/**
 * One process-wide ViewModel that owns the long-lived collaborators
 * (IdentityStore, PeerRegistry, PairingController). Activity recreation
 * doesn't restart the handshake socket.
 *
 * The view layer is presentational: it observes [pairingState] +
 * [pairingPayload] and calls back into the methods here on user action.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    val identityStore = IdentityStore(application)
    val peerRegistry = PeerRegistry(application)
    val pairing = PairingController(identityStore, peerRegistry)
    val files = FileTransferManager(application)
    private val settings = SettingsStore(application)

    init {
        // Bind file-transfer manager to the SecureChannel the moment
        // pairing completes; unbind on cancel / failure so a fresh attempt
        // starts with a clean transfer table.
        viewModelScope.launch {
            pairing.state.collect { st ->
                when (st) {
                    is PairingController.State.Ready -> {
                        SessionHolder.get()?.let(files::bind)
                    }
                    is PairingController.State.Idle,
                    is PairingController.State.Failed -> files.unbind()
                    else -> Unit
                }
            }
        }
    }

    private val _relayUrl = MutableStateFlow(settings.relayUrl)
    val relayUrl: StateFlow<String> = _relayUrl.asStateFlow()

    /** Generated locally on the controlled side, stable across recomposition. */
    private val _roomCode = MutableStateFlow<String?>(null)
    val roomCode: StateFlow<String?> = _roomCode.asStateFlow()

    fun setRelayUrl(url: String) {
        val trimmed = url.trim()
        _relayUrl.value = trimmed
        settings.relayUrl = trimmed
    }

    fun startPairingAsControlled() {
        val url = _relayUrl.value
        if (!url.startsWith("wss://") && !url.startsWith("ws://")) {
            return  // UI guards this; bail silently if it slipped through
        }
        val code = RoomCode.generate()
        _roomCode.value = code
        val metrics = getApplication<Application>().resources.displayMetrics
        pairing.start(
            role = "controlled", relayUrl = url, roomId = code,
            peerIdPubExpected = null,
            myWidthPx = metrics.widthPixels, myHeightPx = metrics.heightPixels,
        )
    }

    fun startPairingAsControllerFromQr(qrText: String): Boolean {
        val payload = com.paperclip.remote.pair.QrPayload.decode(qrText) ?: return false
        val url = _relayUrl.value
        if (!url.startsWith("wss://") && !url.startsWith("ws://")) return false
        _roomCode.value = payload.roomId
        pairing.start(
            role = "controller", relayUrl = url, roomId = payload.roomId,
            peerIdPubExpected = payload.idPub,
        )
        return true
    }

    fun startPairingAsControllerFromCode(code: String): Boolean {
        val normalized = RoomCode.normalize(code) ?: return false
        val url = _relayUrl.value
        if (!url.startsWith("wss://") && !url.startsWith("ws://")) return false
        _roomCode.value = normalized
        // No QR → no fingerprint to validate against; the safety-code
        // step on screen is the only MITM defense in this path.
        pairing.start(
            role = "controller", relayUrl = url, roomId = normalized,
            peerIdPubExpected = null,
        )
        return true
    }

    fun confirmPairing() = pairing.confirm()
    fun cancelPairing() {
        pairing.cancel()
        _roomCode.value = null
    }

    /** Controller-side: ship a tap to the controlled peer. */
    fun sendTap(peerX: Int, peerY: Int) {
        SessionHolder.get()?.sendText(ControlMessage.Tap(peerX, peerY, System.currentTimeMillis()))
    }

    /** Controller-side: ship a swipe to the controlled peer. */
    fun sendSwipe(x1: Int, y1: Int, x2: Int, y2: Int, durMs: Int) {
        SessionHolder.get()?.sendText(
            ControlMessage.Swipe(x1, y1, x2, y2, durMs, System.currentTimeMillis())
        )
    }

    fun sendFile(uri: Uri, displayName: String, mime: String) {
        files.sendUri(uri, displayName, mime)
    }

    override fun onCleared() {
        pairing.cancel()
        files.release()
        super.onCleared()
    }
}
