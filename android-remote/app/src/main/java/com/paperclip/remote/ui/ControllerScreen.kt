package com.paperclip.remote.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
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
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.paperclip.remote.MainViewModel
import com.paperclip.remote.pair.PairingController
import com.paperclip.remote.playback.VideoSurface

@Composable
fun ControllerScreen(onScanQr: () -> Unit = {}) {
    val vm: MainViewModel = viewModel()
    val state by vm.pairing.state.collectAsStateWithLifecycle()
    val relayUrl by vm.relayUrl.collectAsStateWithLifecycle()

    var code by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }

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
            is PairingController.State.Idle, is PairingController.State.Failed -> {
                Button(
                    onClick = onScanQr,
                    enabled = relayUrl.startsWith("ws"),
                ) { Text("Scan QR") }
                Text("— or —", modifier = Modifier.padding(vertical = 8.dp))
                OutlinedTextField(
                    value = code,
                    onValueChange = { code = it.uppercase() },
                    label = { Text("Room code (6 chars)") },
                    singleLine = true,
                )
                Button(
                    onClick = {
                        val ok = vm.startPairingAsControllerFromCode(code)
                        error = if (!ok) "Check the room code and that the relay URL is set." else null
                    },
                    enabled = relayUrl.startsWith("ws") && code.length == 6,
                    modifier = Modifier.padding(top = 12.dp),
                ) { Text("Connect") }
                error?.let { Text(it, modifier = Modifier.padding(top = 8.dp)) }
                if (s is PairingController.State.Failed) {
                    Text("Last attempt: ${s.reason}", modifier = Modifier.padding(top = 16.dp))
                }
            }
            is PairingController.State.Connecting -> Text("Connecting to relay…")
            is PairingController.State.AwaitingPeerHello -> Text("Waiting for the other phone…")
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
                Text("Paired — drag or tap to control the other phone.",
                     modifier = Modifier.padding(bottom = 12.dp))
                val aspect = s.peerWidthPx.toFloat() / s.peerHeightPx
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(aspect),
                ) {
                    VideoSurface(
                        peerWidth = s.peerWidthPx,
                        peerHeight = s.peerHeightPx,
                        modifier = Modifier.fillMaxSize(),
                        onTap = vm::sendTap,
                        onSwipe = vm::sendSwipe,
                    )
                }
                HardwareKeyRow()
                QualityControlSection()
                TextEntrySection()
                FileTransferSection()
                OutlinedButton(onClick = vm::cancelPairing,
                               modifier = Modifier.padding(top = 16.dp)) {
                    Text("Disconnect")
                }
            }
        }
    }
}
