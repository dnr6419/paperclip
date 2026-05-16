package com.paperclip.remote

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.paperclip.remote.ui.ControlledScreen
import com.paperclip.remote.ui.ControllerScreen
import com.paperclip.remote.ui.HomeScreen
import com.paperclip.remote.ui.QrScannerScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface { App() }
            }
        }
    }
}

@androidx.compose.runtime.Composable
private fun App() {
    val nav = rememberNavController()
    NavHost(navController = nav, startDestination = "home") {
        composable("home") {
            HomeScreen(
                onChooseController = { nav.navigate("controller") },
                onChooseControlled = { nav.navigate("controlled") },
            )
        }
        composable("controller") {
            ControllerScreen(onScanQr = { nav.navigate("scan") })
        }
        composable("controlled") { ControlledScreen() }
        composable("scan") {
            val vm: MainViewModel = viewModel()
            QrScannerScreen(
                onScanned = { text ->
                    if (vm.startPairingAsControllerFromQr(text)) {
                        nav.popBackStack("controller", inclusive = false)
                    }
                },
                onCancel = { nav.popBackStack() },
            )
        }
    }
}
