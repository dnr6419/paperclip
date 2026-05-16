package com.paperclip.remote.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.paperclip.remote.R

@Composable
fun HomeScreen(
    onChooseController: () -> Unit,
    onChooseControlled: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("paperclip remote")
        Button(onClick = onChooseController, modifier = Modifier.padding(top = 32.dp)) {
            Text(stringResource(R.string.role_controller))
        }
        Button(onClick = onChooseControlled, modifier = Modifier.padding(top = 16.dp)) {
            Text(stringResource(R.string.role_controlled))
        }
    }
}
