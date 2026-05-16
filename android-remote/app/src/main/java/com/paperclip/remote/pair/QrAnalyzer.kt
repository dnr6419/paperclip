package com.paperclip.remote.pair

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.zxing.BarcodeFormat
import com.google.zxing.BinaryBitmap
import com.google.zxing.DecodeHintType
import com.google.zxing.MultiFormatReader
import com.google.zxing.NotFoundException
import com.google.zxing.PlanarYUVLuminanceSource
import com.google.zxing.common.HybridBinarizer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * CameraX ImageAnalysis.Analyzer that scans the Y plane of each frame
 * for a QR code and reports the decoded text exactly once via
 * [onResult]. After a successful decode the analyzer goes silent —
 * caller is responsible for tearing the camera down.
 */
class QrAnalyzer(
    private val onResult: (String) -> Unit,
) : ImageAnalysis.Analyzer {

    private val reader = MultiFormatReader().apply {
        setHints(mapOf(
            DecodeHintType.POSSIBLE_FORMATS to listOf(BarcodeFormat.QR_CODE),
            DecodeHintType.TRY_HARDER to true,
        ))
    }
    private val done = AtomicBoolean(false)

    override fun analyze(image: ImageProxy) {
        if (done.get()) {
            image.close()
            return
        }
        try {
            val plane = image.planes[0]
            val rowStride = plane.rowStride
            val width = image.width
            val height = image.height
            val buf = plane.buffer
            // Copy the Y plane row by row in case rowStride > width.
            val data = ByteArray(width * height)
            var dst = 0
            val tmp = ByteArray(rowStride)
            buf.rewind()
            for (y in 0 until height) {
                buf.position(y * rowStride)
                val toRead = minOf(rowStride, buf.remaining())
                buf.get(tmp, 0, toRead)
                System.arraycopy(tmp, 0, data, dst, width)
                dst += width
            }
            val source = PlanarYUVLuminanceSource(
                data, width, height, 0, 0, width, height, false,
            )
            val result = reader.decode(BinaryBitmap(HybridBinarizer(source)))
            if (done.compareAndSet(false, true)) {
                onResult(result.text)
            }
        } catch (_: NotFoundException) {
            // No QR code in this frame; keep scanning.
        } catch (_: Throwable) {
            // Any other decoder error: drop the frame, keep going.
        } finally {
            reader.reset()
            image.close()
        }
    }
}
