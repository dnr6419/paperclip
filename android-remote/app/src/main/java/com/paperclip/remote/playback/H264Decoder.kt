package com.paperclip.remote.playback

import android.media.MediaCodec
import android.media.MediaFormat
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * H.264 Annex-B decoder rendering directly to a Surface. Designed to be
 * fed access units exactly as the encoder emits them on the other side
 * (CSD prepended to keyframes, raw NALs otherwise).
 *
 * The decoder bootstraps off the first access unit, which must be a
 * keyframe and must therefore include SPS+PPS. Subsequent feeds are
 * appended; if input arrives faster than the decoder can consume, the
 * pending queue is bounded — older non-keyframes drop first to preserve
 * latency over completeness, matching PROTOCOL.md's backpressure rule.
 */
class H264Decoder(
    private val outputSurface: Surface,
    private val width: Int,
    private val height: Int,
    private val onError: (Throwable) -> Unit = {},
) {
    private val codec: MediaCodec = MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
    private val pendingInput = ConcurrentLinkedQueue<ByteArray>()
    private var configured = false
    @Volatile private var running = false

    fun start() {
        running = true
        codec.setCallback(Callback())
    }

    /** Feed one Annex-B access unit (may include SPS+PPS prefix). */
    fun feed(accessUnit: ByteArray) {
        if (!running) return
        if (pendingInput.size > MAX_PENDING) pendingInput.poll()
        pendingInput.add(accessUnit)
        ensureConfigured(accessUnit)
    }

    private fun ensureConfigured(firstAu: ByteArray) {
        if (configured) return
        // The first frame must be a keyframe carrying CSD. Hand it to
        // MediaCodec via the format directly so the codec configures
        // before the first input buffer.
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setByteBuffer("csd-0", ByteBuffer.wrap(firstAu))
        }
        try {
            codec.configure(format, outputSurface, null, 0)
            codec.start()
            configured = true
        } catch (t: Throwable) {
            Log.e(TAG, "decoder configure failed", t)
            onError(t)
        }
    }

    fun stop() {
        running = false
        try { codec.stop() } catch (_: Exception) {}
        try { codec.release() } catch (_: Exception) {}
    }

    private inner class Callback : MediaCodec.Callback() {
        override fun onInputBufferAvailable(c: MediaCodec, index: Int) {
            if (!running) return
            val au = pendingInput.poll() ?: return run {
                // Return the buffer immediately with empty payload; the
                // codec will ask for another one shortly.
                c.queueInputBuffer(index, 0, 0, 0L, 0)
            }
            val buf = c.getInputBuffer(index) ?: return
            buf.clear()
            buf.put(au)
            c.queueInputBuffer(index, 0, au.size, System.nanoTime() / 1000, 0)
        }

        override fun onOutputBufferAvailable(c: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
            c.releaseOutputBuffer(index, info.size > 0)
        }

        override fun onError(c: MediaCodec, e: MediaCodec.CodecException) {
            Log.e(TAG, "decoder error", e)
            onError(e)
        }

        override fun onOutputFormatChanged(c: MediaCodec, format: MediaFormat) {
            // No-op: rendering to a Surface, so we don't pull raw frames.
        }
    }

    companion object {
        private const val MAX_PENDING = 4
        private const val TAG = "H264Decoder"
    }
}
