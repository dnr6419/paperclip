package com.paperclip.remote.playback

import android.view.Surface
import com.paperclip.remote.session.SessionHolder
import com.paperclip.remote.transport.BinaryTag
import com.paperclip.remote.transport.SecureChannel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.launch

/**
 * Drives an [H264Decoder] from the active SecureChannel's incoming
 * binary stream. Attach to a Surface at the moment SurfaceView is
 * ready; detach in onPause / onStop / surfaceDestroyed.
 *
 * Holds no codec while detached — recreating the decoder on each attach
 * keeps the MediaCodec state machine simple and lets the controlled
 * side's next IDR re-seed without us having to manage flush() correctly.
 */
class VideoPlayer(
    private val width: Int,
    private val height: Int,
    private val onError: (Throwable) -> Unit = {},
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default),
) {
    private var decoder: H264Decoder? = null
    private var pump: Job? = null

    fun attach(surface: Surface) {
        detach()
        val dec = H264Decoder(outputSurface = surface, width = width, height = height, onError = onError)
        dec.start()
        decoder = dec
        pump = scope.launch {
            while (true) {
                val channel = SessionHolder.get()
                if (channel == null) {
                    delay(200)
                    continue
                }
                consume(channel, dec)
                // SessionHolder cleared -> SecureChannel torn down; loop and wait.
                delay(200)
            }
        }
    }

    fun detach() {
        pump?.cancel()
        pump = null
        decoder?.stop()
        decoder = null
    }

    fun release() {
        detach()
        scope.cancel()
    }

    private suspend fun consume(channel: SecureChannel, dec: H264Decoder) {
        channel.incoming
            .filterIsInstance<SecureChannel.Plain.Binary>()
            .collect { binary ->
                if (binary.tag == BinaryTag.VIDEO) {
                    dec.feed(binary.payload)
                }
            }
    }
}
