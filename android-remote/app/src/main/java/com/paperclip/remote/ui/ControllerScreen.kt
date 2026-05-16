package com.paperclip.remote.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.paperclip.remote.pair.RoomCode

@Composable
fun ControllerScreen() {
    var relayUrl by remember { mutableStateOf("wss://") }
    var code by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        OutlinedTextField(
            value = relayUrl,
            onValueChange = { relayUrl = it },
            label = { Text("Relay URL") },
            modifier = Modifier.padding(bottom = 12.dp),
        )
        OutlinedTextField(
            value = code,
            onValueChange = { code = it.uppercase() },
            label = { Text("Room code") },
        )
        Button(
            onClick = {
                val normalized = RoomCode.normalize(code)
                error = if (normalized == null) {
                    "Code must be 6 valid base32 characters."
                } else if (!relayUrl.startsWith("wss://")) {
                    "Relay URL must start with wss://"
                } else {
                    // Actual connect happens in a follow-up commit that adds
                    // the video surface + RelayClient wiring.
                    null
                }
            },
            modifier = Modifier.padding(top = 16.dp),
        ) { Text("Connect") }
        error?.let { Text(it, modifier = Modifier.padding(top = 8.dp)) }
    }
}
