package com.paperclip.remote.input

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Path
import android.os.Bundle
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import com.paperclip.remote.session.SessionHolder
import com.paperclip.remote.transport.ControlMessage
import com.paperclip.remote.transport.SecureChannel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.launch

/**
 * AccessibilityService that translates incoming [ControlMessage.Tap],
 * [ControlMessage.Swipe], and [ControlMessage.Key] frames into actual
 * gestures on the controlled phone. Hooks into the active SecureChannel
 * via [SessionHolder].
 *
 * The user must enable the service once from
 *   Settings → Accessibility → Paperclip Remote → On
 * before any input is delivered. The app surfaces a deep-link prompt
 * the first time a controller tries to send a tap with no service
 * enabled.
 *
 * Coordinate space matches what the controller sends, which per
 * PROTOCOL.md is the controlled phone's pixel space — the controller is
 * responsible for the mapping (it has both displays' geometry from the
 * exchanged hellos).
 */
class RemoteInputService : AccessibilityService() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var subscription: Job? = null

    override fun onServiceConnected() {
        super.onServiceConnected()
        Log.i(TAG, "service connected")
        // Re-attach every time the service is bound. SessionHolder.get()
        // may be null at this moment; we observe its changes via the
        // existing SecureChannel.incoming when a session shows up.
        attachToSession()
    }

    override fun onInterrupt() {
        Log.w(TAG, "service interrupted")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // Not interested in inspecting node tree events; we only emit.
    }

    override fun onUnbind(intent: android.content.Intent?): Boolean {
        subscription?.cancel()
        subscription = null
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    private fun attachToSession() {
        subscription?.cancel()
        // Poll-then-collect: when a session arrives later, the next loop
        // iteration picks it up. We don't need a Flow on SessionHolder
        // because the lifetime is short and there is no churn.
        subscription = scope.launch {
            while (true) {
                val channel = SessionHolder.get()
                if (channel != null) {
                    consume(channel)
                }
                // Service is unbound or session ended; idle 250ms and re-check.
                kotlinx.coroutines.delay(250)
            }
        }
    }

    private suspend fun consume(channel: SecureChannel) {
        channel.incoming
            .filterIsInstance<SecureChannel.Plain.Text>()
            .collectLatest { plain ->
                when (val msg = plain.message) {
                    is ControlMessage.Tap       -> doTap(msg.x.toFloat(), msg.y.toFloat())
                    is ControlMessage.Swipe     -> doSwipe(msg.x1.toFloat(), msg.y1.toFloat(),
                                                           msg.x2.toFloat(), msg.y2.toFloat(),
                                                           msg.ms.toLong().coerceAtLeast(50L))
                    is ControlMessage.Key       -> doKey(msg.code)
                    is ControlMessage.Type      -> doType(msg.text)
                    is ControlMessage.Clipboard -> doClipboard(msg.text)
                    else -> { /* not an input frame */ }
                }
            }
    }

    private fun doTap(x: Float, y: Float) {
        val path = Path().apply { moveTo(x, y) }
        val stroke = GestureDescription.StrokeDescription(path, 0L, 50L)
        dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
    }

    private fun doSwipe(x1: Float, y1: Float, x2: Float, y2: Float, durMs: Long) {
        val path = Path().apply { moveTo(x1, y1); lineTo(x2, y2) }
        val stroke = GestureDescription.StrokeDescription(path, 0L, durMs)
        dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
    }

    private fun doKey(code: String) {
        val action = when (code.uppercase()) {
            "BACK"    -> GLOBAL_ACTION_BACK
            "HOME"    -> GLOBAL_ACTION_HOME
            "RECENTS" -> GLOBAL_ACTION_RECENTS
            "NOTIFICATIONS" -> GLOBAL_ACTION_NOTIFICATIONS
            else -> return
        }
        performGlobalAction(action)
    }

    private fun doType(text: String) {
        val root = rootInActiveWindow ?: run {
            Log.d(TAG, "type: no active window"); return
        }
        val focused = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) ?: run {
            Log.d(TAG, "type: no focused input"); return
        }
        val args = Bundle().apply {
            putCharSequence(
                AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                text,
            )
        }
        val ok = focused.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
        if (!ok) Log.w(TAG, "type: ACTION_SET_TEXT rejected by node")
    }

    private fun doClipboard(text: String) {
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager ?: return
        try {
            cm.setPrimaryClip(ClipData.newPlainText("paperclip-remote", text))
        } catch (e: SecurityException) {
            // Android Q+ blocks background clipboard writes; documented caveat.
            Log.w(TAG, "clipboard: write blocked (app not foreground)", e)
        }
    }

    companion object {
        private const val TAG = "RemoteInputService"
    }
}
