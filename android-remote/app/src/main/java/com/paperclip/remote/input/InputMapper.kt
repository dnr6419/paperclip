package com.paperclip.remote.input

/**
 * Maps a touch event in the controller's video view back to a pixel
 * coordinate on the controlled phone's display.
 *
 * The controller sees the controlled phone's screen via H.264. Two
 * displays usually have different sizes and possibly different aspect
 * ratios, so we letterbox (FIT) by default — taps outside the drawn
 * video region are dropped instead of being clamped to an edge that
 * never had any content under the user's finger.
 *
 * Pure-logic class with no Android dependencies; covered by JVM unit
 * tests in `docs/_crosscheck/InputMapperCheck.kt`.
 */
class InputMapper(
    private val peerWidthPx: Int,
    private val peerHeightPx: Int,
    private val mode: Mode = Mode.FIT,
) {
    enum class Mode { FIT, STRETCH }

    /** Returns null when the (viewX, viewY) falls outside the drawn video region (FIT only). */
    fun mapPoint(viewX: Float, viewY: Float, viewW: Int, viewH: Int): Pair<Int, Int>? {
        if (viewW <= 0 || viewH <= 0 || peerWidthPx <= 0 || peerHeightPx <= 0) return null
        return when (mode) {
            Mode.STRETCH -> stretch(viewX, viewY, viewW, viewH)
            Mode.FIT     -> fit(viewX, viewY, viewW, viewH)
        }
    }

    private fun stretch(viewX: Float, viewY: Float, viewW: Int, viewH: Int): Pair<Int, Int> {
        val px = (viewX / viewW * peerWidthPx).toInt().coerceIn(0, peerWidthPx - 1)
        val py = (viewY / viewH * peerHeightPx).toInt().coerceIn(0, peerHeightPx - 1)
        return px to py
    }

    private fun fit(viewX: Float, viewY: Float, viewW: Int, viewH: Int): Pair<Int, Int>? {
        val viewAspect = viewW.toFloat() / viewH
        val peerAspect = peerWidthPx.toFloat() / peerHeightPx
        val drawnW: Int; val drawnH: Int; val offX: Int; val offY: Int
        if (viewAspect > peerAspect) {
            // Letterbox left/right: video has full height, narrower than view.
            drawnH = viewH
            drawnW = (viewH * peerAspect).toInt()
            offX = (viewW - drawnW) / 2
            offY = 0
        } else {
            // Letterbox top/bottom: video has full width, shorter than view.
            drawnW = viewW
            drawnH = (viewW / peerAspect).toInt()
            offX = 0
            offY = (viewH - drawnH) / 2
        }
        if (viewX < offX || viewX > offX + drawnW ||
            viewY < offY || viewY > offY + drawnH) return null
        val rx = (viewX - offX) / drawnW * peerWidthPx
        val ry = (viewY - offY) / drawnH * peerHeightPx
        return rx.toInt().coerceIn(0, peerWidthPx - 1) to
               ry.toInt().coerceIn(0, peerHeightPx - 1)
    }
}
