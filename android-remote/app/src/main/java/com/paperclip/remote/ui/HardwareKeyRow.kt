package com.paperclip.remote.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.paperclip.remote.MainViewModel

/**
 * BACK / HOME / RECENTS / NOTIFICATIONS chip row shown above the video
 * surface on the controller side. Maps to `ControlMessage.Key` which
 * RemoteInputService translates to `performGlobalAction` on the
 * controlled phone.
 */
@Composable
fun HardwareKeyRow(modifier: Modifier = Modifier) {
    val vm: MainViewModel = viewModel()
    Row(
        modifier = modifier.fillMaxWidth().padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        for ((label, code) in keys) {
            AssistChip(
                onClick = { vm.sendKey(code) },
                label = { Text(label) },
                colors = AssistChipDefaults.assistChipColors(),
                modifier = Modifier.weight(1f),
            )
        }
    }
}

private val keys = listOf(
    "◀ Back"     to "BACK",
    "● Home"     to "HOME",
    "▢ Recents"  to "RECENTS",
    "🔔"         to "NOTIFICATIONS",
)
