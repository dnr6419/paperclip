package com.paperclip.remote.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.paperclip.remote.MainViewModel

/**
 * Two small entry points exposed in the Ready state on each side:
 *
 *  - "Type…" sends text via `ControlMessage.Type` → AccessibilityService
 *    fills the currently focused EditText on the peer.
 *  - "Push clipboard" reads this phone's clipboard and ships it as
 *    `ControlMessage.Clipboard`. Receiver writes to its clipboard; on
 *    Android Q+ this may silently fail if the receiver app isn't
 *    foreground (documented caveat in PROTOCOL.md).
 *
 * The screens that include this section are role-aware: the controller
 * shows both buttons (typing is the controller-→-controlled direction);
 * the controlled-only screen shows just the clipboard push.
 */
@Composable
fun TextEntrySection(modifier: Modifier = Modifier) {
    val vm: MainViewModel = viewModel()
    var draft by remember { mutableStateOf("") }
    Column(modifier = modifier.fillMaxWidth().padding(top = 16.dp),
           verticalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(
            value = draft,
            onValueChange = { draft = it },
            label = { Text("Type text to send to the other phone") },
            singleLine = false,
            modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(
                onClick = { vm.sendType(draft); draft = "" },
                enabled = draft.isNotEmpty(),
            ) { Text("Type") }
            OutlinedButton(onClick = { vm.sendMyClipboard() }) {
                Text("Push clipboard")
            }
        }
    }
}

@Composable
fun ClipboardOnlySection(modifier: Modifier = Modifier) {
    val vm: MainViewModel = viewModel()
    OutlinedButton(
        onClick = { vm.sendMyClipboard() },
        modifier = modifier.padding(top = 12.dp),
    ) { Text("Push my clipboard to the other phone") }
}
