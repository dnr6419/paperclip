package com.paperclip.remote.ui

import android.app.Activity
import android.content.Context
import android.media.projection.MediaProjectionManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.paperclip.remote.MainViewModel
import com.paperclip.remote.capture.ScreenCaptureService
import com.paperclip.remote.pair.PairingController
import com.paperclip.remote.pair.QrBitmap
import com.paperclip.remote.pair.QrPayload

@Composable
fun ControlledScreen() {
    val vm: MainViewModel = viewModel()
    val state by vm.pairing.state.collectAsStateWithLifecycle()
    val qrPayload by vm.pairing.qrPayload.collectAsStateWithLifecycle()
    val roomCode by vm.roomCode.collectAsStateWithLifecycle()
    val relayUrl by vm.relayUrl.collectAsStateWithLifecycle()
    val context = LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        OutlinedTextField(
            value = relayUrl,
            onValueChange = vm::setRelayUrl,
            label = { Text("Relay URL (wss://…)") },
            singleLine = true,
            enabled = state is PairingController.State.Idle || state is PairingController.State.Failed,
            modifier = Modifier.padding(bottom = 16.dp),
        )

        when (val s = state) {
            is PairingController.State.Idle -> {
                Button(onClick = vm::startPairingAsControlled,
                       enabled = relayUrl.startsWith("ws")) {
                    Text("Start pairing")
                }
            }
            is PairingController.State.Connecting -> {
                Text("Connecting to relay…")
            }
            is PairingController.State.AwaitingPeerHello -> {
                roomCode?.let {
                    Text("Room code", modifier = Modifier.padding(bottom = 4.dp))
                    Text(it, fontSize = 40.sp, fontFamily = FontFamily.Monospace)
                }
                Spacer(Modifier.height(16.dp))
                qrPayload?.let { payload ->
                    val text = QrPayload.encode(payload)
                    Image(
                        bitmap = QrBitmap.render(text, 600),
                        contentDescription = "Pairing QR",
                        modifier = Modifier.size(260.dp).padding(8.dp),
                    )
                }
                Text("Scan this on the other phone, or type the code.",
                     modifier = Modifier.padding(top = 8.dp))
                OutlinedButton(onClick = vm::cancelPairing,
                               modifier = Modifier.padding(top = 16.dp)) {
                    Text("Cancel")
                }
            }
            is PairingController.State.AwaitingConfirm -> {
                Text("Safety code", modifier = Modifier.padding(bottom = 4.dp))
                Text(s.safetyCode, fontSize = 24.sp, fontFamily = FontFamily.Monospace,
                     textAlign = TextAlign.Center)
                Text(
                    if (s.isResumed) "(known peer — auto-confirmable)"
                    else "Confirm both phones show the same code, then accept.",
                    modifier = Modifier.padding(top = 8.dp), textAlign = TextAlign.Center,
                )
                Row(modifier = Modifier.padding(top = 16.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedButton(onClick = vm::cancelPairing) { Text("Cancel") }
                    Button(onClick = vm::confirmPairing) { Text("Accept") }
                }
            }
            is PairingController.State.Ready -> {
                Text("Paired " + (if (s.isResumed) "(resumed)" else "(first time)"),
                     modifier = Modifier.padding(bottom = 16.dp))
                ShareScreenSection(context = context)
            }
            is PairingController.State.Failed -> {
                Text("Failed: ${s.reason}", modifier = Modifier.padding(bottom = 16.dp))
                Button(onClick = vm::cancelPairing) { Text("Reset") }
            }
        }
    }
}

@Composable
private fun ShareScreenSection(context: Context) {
    var sharing by remember { mutableStateOf(false) }
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult(),
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            ScreenCaptureService.start(context, result.resultCode, result.data!!)
            sharing = true
        }
    }
    Button(
        onClick = {
            val mgr = context.getSystemService(Context.MEDIA_PROJECTION_SERVICE)
                    as MediaProjectionManager
            launcher.launch(mgr.createScreenCaptureIntent())
        },
        enabled = !sharing,
    ) {
        Text(if (sharing) "Sharing…" else "Start sharing")
    }
}
