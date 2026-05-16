package com.paperclip.remote.playback

import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.paperclip.remote.input.InputMapper

/**
 * Composable SurfaceView that hosts the H.264 playback path and routes
 * single-finger taps + drags through [InputMapper] back to the peer's
 * pixel coordinates. The caller plumbs those out to a SecureChannel send.
 */
@Composable
fun VideoSurface(
    peerWidth: Int,
    peerHeight: Int,
    modifier: Modifier = Modifier,
    onTap: (peerX: Int, peerY: Int) -> Unit,
    onSwipe: (px1: Int, py1: Int, px2: Int, py2: Int, durMs: Int) -> Unit,
) {
    val mapper = remember(peerWidth, peerHeight) {
        InputMapper(peerWidth, peerHeight, InputMapper.Mode.FIT)
    }
    val player = remember(peerWidth, peerHeight) {
        VideoPlayer(width = peerWidth, height = peerHeight)
    }
    DisposableEffect(player) { onDispose { player.release() } }

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            val view = SurfaceView(ctx)
            view.holder.addCallback(object : SurfaceHolder.Callback {
                override fun surfaceCreated(h: SurfaceHolder) { player.attach(h.surface) }
                override fun surfaceChanged(h: SurfaceHolder, f: Int, w: Int, ht: Int) {}
                override fun surfaceDestroyed(h: SurfaceHolder) { player.detach() }
            })
            // Single-touch gesture handling. Multi-touch (pinch, two-finger
            // scroll) lands separately; the MVP only needs tap + swipe.
            val gestureState = SingleTouchState()
            view.setOnTouchListener { v: View, ev: MotionEvent ->
                when (ev.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        gestureState.start(ev.x, ev.y, ev.eventTime)
                        true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        gestureState.update(ev.x, ev.y)
                        true
                    }
                    MotionEvent.ACTION_UP -> {
                        gestureState.finish(ev.x, ev.y, ev.eventTime,
                            viewW = v.width, viewH = v.height,
                            mapper = mapper, onTap = onTap, onSwipe = onSwipe)
                        true
                    }
                    MotionEvent.ACTION_CANCEL -> {
                        gestureState.reset(); true
                    }
                    else -> false
                }
            }
            view
        },
    )
}

/**
 * Distinguishes a tap (negligible move, short duration) from a swipe
 * (any meaningful displacement) on a single finger.
 */
private class SingleTouchState {
    private var x0 = 0f; private var y0 = 0f; private var t0 = 0L
    private var xMax = 0f; private var yMax = 0f
    private var moved = false

    fun start(x: Float, y: Float, t: Long) {
        x0 = x; y0 = y; t0 = t
        xMax = x; yMax = y; moved = false
    }

    fun update(x: Float, y: Float) {
        if (kotlin.math.abs(x - x0) > MOVE_THRESHOLD_PX ||
            kotlin.math.abs(y - y0) > MOVE_THRESHOLD_PX) {
            moved = true
        }
        xMax = x; yMax = y
    }

    fun finish(
        x: Float, y: Float, t: Long, viewW: Int, viewH: Int,
        mapper: InputMapper,
        onTap: (Int, Int) -> Unit,
        onSwipe: (Int, Int, Int, Int, Int) -> Unit,
    ) {
        if (!moved) {
            val pt = mapper.mapPoint(x, y, viewW, viewH) ?: return
            onTap(pt.first, pt.second)
        } else {
            val start = mapper.mapPoint(x0, y0, viewW, viewH) ?: return
            val end = mapper.mapPoint(x, y, viewW, viewH) ?: return
            onSwipe(start.first, start.second, end.first, end.second,
                    (t - t0).coerceAtLeast(50L).toInt())
        }
        reset()
    }

    fun reset() { moved = false }

    companion object {
        private const val MOVE_THRESHOLD_PX = 12f
    }
}
