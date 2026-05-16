package com.paperclip.remote.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.paperclip.remote.pair.RoomCode

/**
 * Generates a one-time room code, shows it big, waits for a controller.
 *
 * The QR + actual connect flow ship with the capture service in a
 * follow-up commit — surfacing the code at all is the smallest useful
 * step the user can react to right now.
 */
@Composable
fun ControlledScreen() {
    val code = remember { RoomCode.generate() }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Room code", modifier = Modifier.padding(bottom = 16.dp))
        Text(
            text = code,
            fontSize = 56.sp,
            textAlign = TextAlign.Center,
        )
        Text(
            text = "Open the other phone and enter this code.",
            modifier = Modifier.padding(top = 24.dp),
            textAlign = TextAlign.Center,
        )
    }
}
