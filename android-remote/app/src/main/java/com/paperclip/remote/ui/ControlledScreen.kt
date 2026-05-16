package com.paperclip.remote.ui

import android.app.Activity
import android.content.Context
import android.media.projection.MediaProjectionManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.paperclip.remote.capture.ScreenCaptureService
import com.paperclip.remote.pair.RoomCode

@Composable
fun ControlledScreen() {
    val code = remember { RoomCode.generate() }
    val context = LocalContext.current
    var sharing by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult(),
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            ScreenCaptureService.start(context, result.resultCode, result.data!!)
            sharing = true
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Room code", modifier = Modifier.padding(bottom = 16.dp))
        Text(text = code, fontSize = 56.sp, textAlign = TextAlign.Center)
        Text(
            text = "Open the other phone and enter this code.",
            modifier = Modifier.padding(top = 24.dp),
            textAlign = TextAlign.Center,
        )
        Button(
            onClick = {
                val mgr = context.getSystemService(Context.MEDIA_PROJECTION_SERVICE)
                        as MediaProjectionManager
                launcher.launch(mgr.createScreenCaptureIntent())
            },
            enabled = !sharing,
            modifier = Modifier.padding(top = 32.dp),
        ) {
            Text(if (sharing) "Sharing…" else "Start sharing")
        }
    }
}
