package com.paperclip.remote.capture

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer

/**
 * H.264 surface-input encoder. The caller passes a Surface backed by
 * MediaProjection's VirtualDisplay; the encoder reads frames from it
 * and emits Annex-B byte-stream NAL access units via [onAccessUnit].
 *
 * The first access unit (CSD: SPS+PPS) is cached and re-prepended to
 * every keyframe we emit so that a controller joining mid-stream — or
 * recovering from a packet drop — can decode without an out-of-band CSD
 * exchange.
 */
class H264Encoder(
    private val width: Int,
    private val height: Int,
    private val bitrate: Int = 4_000_000,
    private val fps: Int = 30,
    private val keyframeIntervalSec: Int = 2,
    private val onAccessUnit: (ByteArray, isKeyFrame: Boolean) -> Unit,
) {
    private val codec: MediaCodec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
    val inputSurface: Surface

    /** SPS+PPS cached from the first BUFFER_FLAG_CODEC_CONFIG output. */
    private var csd: ByteArray? = null

    init {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, keyframeIntervalSec)
            setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
        }
        codec.setCallback(Callback())
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        inputSurface = codec.createInputSurface()
    }

    fun start() {
        codec.start()
    }

    fun stop() {
        try { codec.stop() } catch (_: Exception) {}
        try { codec.release() } catch (_: Exception) {}
        try { inputSurface.release() } catch (_: Exception) {}
    }

    /** Ask the encoder for an IDR on the next frame (e.g. after a quality change). */
    fun requestKeyFrame() {
        codec.setParameters(android.os.Bundle().apply {
            putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0)
        })
    }

    fun updateBitrate(bps: Int) {
        codec.setParameters(android.os.Bundle().apply {
            putInt(MediaCodec.PARAMETER_KEY_VIDEO_BITRATE, bps)
        })
    }

    private inner class Callback : MediaCodec.Callback() {
        override fun onInputBufferAvailable(c: MediaCodec, index: Int) {
            // No-op: input is a Surface, frames arrive without buffer dance.
        }

        override fun onOutputBufferAvailable(c: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
            val buffer: ByteBuffer = c.getOutputBuffer(index) ?: run {
                c.releaseOutputBuffer(index, false); return
            }
            buffer.position(info.offset)
            buffer.limit(info.offset + info.size)

            val isConfig = (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0
            val isKey = (info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0

            val payload = ByteArray(info.size)
            buffer.get(payload)

            if (isConfig) {
                csd = payload
            } else if (info.size > 0) {
                val output = if (isKey) (csd ?: ByteArray(0)) + payload else payload
                onAccessUnit(output, isKey)
            }
            c.releaseOutputBuffer(index, false)
        }

        override fun onError(c: MediaCodec, e: MediaCodec.CodecException) {
            Log.e(TAG, "encoder error", e)
        }

        override fun onOutputFormatChanged(c: MediaCodec, format: MediaFormat) {
            // Some devices deliver SPS/PPS via the format change instead of
            // the CODEC_CONFIG flag. Concatenate explicitly rather than
            // relying on ByteBuffer position arithmetic.
            val sps = format.getByteBuffer("csd-0")
            val pps = format.getByteBuffer("csd-1")
            if (sps != null && pps != null) {
                val spsBytes = ByteArray(sps.remaining()).also { sps.get(it) }
                val ppsBytes = ByteArray(pps.remaining()).also { pps.get(it) }
                csd = spsBytes + ppsBytes
            }
        }
    }

    companion object {
        private const val TAG = "H264Encoder"
    }
}
