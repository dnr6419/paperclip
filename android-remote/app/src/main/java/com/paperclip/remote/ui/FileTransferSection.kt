package com.paperclip.remote.ui

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.paperclip.remote.MainViewModel
import com.paperclip.remote.files.FileTransferManager

@Composable
fun FileTransferSection() {
    val vm: MainViewModel = viewModel()
    val transfers by vm.files.transfers.collectAsStateWithLifecycle()
    val context = LocalContext.current

    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri: Uri? ->
        if (uri != null) {
            val (name, mime) = queryDisplayName(context, uri)
            vm.sendFile(uri, name, mime)
        }
    }

    Column(modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
           verticalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = { picker.launch(arrayOf("*/*")) }) {
            Text("Send file")
        }
        for ((_, state) in transfers.entries.sortedByDescending { it.value.name }) {
            TransferRow(state)
        }
    }
}

@Composable
private fun TransferRow(state: FileTransferManager.State) {
    Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(horizontalArrangement = Arrangement.SpaceBetween,
            modifier = Modifier.fillMaxWidth()) {
            Text(
                text = "${labelFor(state)}  ${state.name}",
                maxLines = 1, overflow = TextOverflow.MiddleEllipsis,
                modifier = Modifier.weight(1f),
            )
            Text(text = "${(state.progress * 100).toInt()}%")
        }
        if (state is FileTransferManager.State.Sending ||
            state is FileTransferManager.State.Receiving) {
            LinearProgressIndicator(
                progress = { state.progress },
                modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
            )
        }
        if (state is FileTransferManager.State.Failed) {
            Text("failed: ${state.reason}",
                 modifier = Modifier.padding(top = 4.dp))
        }
        if (state is FileTransferManager.State.Done && state.path != null) {
            Text("saved to ${state.path}",
                 maxLines = 1, overflow = TextOverflow.MiddleEllipsis,
                 modifier = Modifier.padding(top = 4.dp))
        }
    }
}

private fun labelFor(state: FileTransferManager.State): String = when (state) {
    is FileTransferManager.State.Sending   -> "↑"
    is FileTransferManager.State.Receiving -> "↓"
    is FileTransferManager.State.Done      -> "✓"
    is FileTransferManager.State.Failed    -> "✗"
}

private fun queryDisplayName(context: Context, uri: Uri): Pair<String, String> {
    val mime = context.contentResolver.getType(uri) ?: "application/octet-stream"
    val cursor = context.contentResolver.query(uri, null, null, null, null)
    cursor?.use {
        if (it.moveToFirst()) {
            val idx = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0) return it.getString(idx) to mime
        }
    }
    return (uri.lastPathSegment ?: "file") to mime
}
