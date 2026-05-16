package com.paperclip.remote.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.paperclip.remote.MainViewModel
import kotlinx.coroutines.delay

/**
 * Slider for live bitrate adjustment, controller-side. Debounced so we
 * don't flood the controlled phone with `quality` frames as the user
 * drags. fps/scale stay at sensible defaults; SEED §10 keeps those as
 * Execute-phase decisions and they're not live-tunable in MediaCodec
 * anyway.
 */
@Composable
fun QualityControlSection(modifier: Modifier = Modifier) {
    val vm: MainViewModel = viewModel()
    var bitrateMbps by remember { mutableFloatStateOf(4f) }   // ~4 Mbps default

    LaunchedEffect(bitrateMbps) {
        // Debounce 300 ms so a slow drag doesn't ship 60 frames/s of
        // `quality` messages. Final value lands once the user pauses.
        delay(300)
        vm.sendQuality(bitrateBps = (bitrateMbps * 1_000_000).toInt())
    }

    Column(
        modifier = modifier.fillMaxWidth().padding(top = 16.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Bitrate")
            Text("%.1f Mbps".format(bitrateMbps))
        }
        Slider(
            value = bitrateMbps,
            onValueChange = { bitrateMbps = it },
            valueRange = 1f..10f,
            steps = 17,         // 0.5 Mbps increments
        )
    }
}
