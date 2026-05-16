package com.paperclip.remote.capture

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.IBinder
import android.util.Log
import android.view.WindowManager
import com.paperclip.remote.R
import com.paperclip.remote.session.SessionHolder

/**
 * Foreground service owning the MediaProjection lifecycle.
 *
 * Start with `Intent(ctx, ScreenCaptureService::class.java).putExtras(...)`
 * carrying the result of MediaProjectionManager.createScreenCaptureIntent()
 * back from the activity. The service stays up until [stopSelf] is called
 * by the WebSocket close handler or by the user tapping the notification.
 */
class ScreenCaptureService : Service() {

    private var projection: MediaProjection? = null
    private var encoder: H264Encoder? = null
    private var virtualDisplay: VirtualDisplay? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        intent ?: return START_NOT_STICKY

        startForegroundCompat()

        val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0)
        val resultData: Intent? = intent.getParcelableExtra(EXTRA_RESULT_DATA)
        if (resultCode == 0 || resultData == null) {
            Log.e(TAG, "missing MediaProjection result")
            stopSelf()
            return START_NOT_STICKY
        }

        val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        val proj = mgr.getMediaProjection(resultCode, resultData)
        if (proj == null) {
            Log.e(TAG, "MediaProjection.getMediaProjection returned null")
            stopSelf()
            return START_NOT_STICKY
        }
        projection = proj
        proj.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                Log.i(TAG, "MediaProjection stopped by system")
                stopSelf()
            }
        }, null)

        val (w, h, dpi) = pickDimensions()
        val enc = H264Encoder(width = w, height = h) { au, _ ->
            SessionHolder.sendVideo(au)
        }
        encoder = enc
        enc.start()

        virtualDisplay = proj.createVirtualDisplay(
            "paperclip-remote",
            w, h, dpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            enc.inputSurface, null, null,
        )

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        virtualDisplay?.release()
        encoder?.stop()
        projection?.stop()
        virtualDisplay = null
        encoder = null
        projection = null
    }

    private fun startForegroundCompat() {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.capture_notification_channel),
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
        val notification: Notification = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle(getString(R.string.capture_notification_title))
            .setContentText(getString(R.string.capture_notification_text))
            .setOngoing(true)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun pickDimensions(): Triple<Int, Int, Int> {
        val wm = getSystemService(WINDOW_SERVICE) as WindowManager
        val metrics = wm.currentWindowMetrics
        val bounds = metrics.bounds
        val density = resources.displayMetrics.densityDpi
        // Scale long edge to <= 1080 to stay under the bitrate budget.
        val maxEdge = 1080
        val longEdge = maxOf(bounds.width(), bounds.height())
        val scale = if (longEdge <= maxEdge) 1.0 else maxEdge.toDouble() / longEdge
        val w = (bounds.width() * scale).toInt() and 0xFFFE   // even for H.264
        val h = (bounds.height() * scale).toInt() and 0xFFFE
        return Triple(w, h, density)
    }

    companion object {
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        private const val CHANNEL_ID = "paperclip-remote.capture"
        private const val NOTIFICATION_ID = 0xC4
        private const val TAG = "ScreenCaptureService"

        fun start(context: Context, resultCode: Int, resultData: Intent) {
            val intent = Intent(context, ScreenCaptureService::class.java).apply {
                putExtra(EXTRA_RESULT_CODE, resultCode)
                putExtra(EXTRA_RESULT_DATA, resultData)
            }
            context.startForegroundService(intent)
        }
    }
}
