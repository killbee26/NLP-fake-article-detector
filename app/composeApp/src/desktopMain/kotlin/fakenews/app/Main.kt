package fakenews.app

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.DarkMode
import androidx.compose.material.icons.outlined.LightMode
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState

fun main() = application {
    val vm = remember { AppViewModel() }
    Window(
        onCloseRequest = { vm.close(); exitApplication() },
        title = "Fake News Detector",
        state = rememberWindowState(size = DpSize(1280.dp, 820.dp)),
    ) {
        window.minimumSize = java.awt.Dimension(1100, 720)
        val state by vm.state.collectAsState()
        AppTheme(state.darkTheme) { App(vm, state) }
    }
}

@Composable
fun App(vm: AppViewModel, state: UiState, initialTab: Int = 0) {
    var tab by remember { mutableStateOf(initialTab) }
    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        Surface(tonalElevation = 2.dp) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Fake News Detector", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text("Fine-tuned DistilBERT vs TF-IDF + Logistic Regression · runs fully offline on CPU",
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
                }
                TabRow(selectedTabIndex = tab, modifier = Modifier.width(260.dp), containerColor = MaterialTheme.colorScheme.surface) {
                    Tab(tab == 0, onClick = { tab = 0 }, text = { Text("Analyze") })
                    Tab(tab == 1, onClick = { tab = 1 }, text = { Text("Model report") })
                }
                Spacer(Modifier.width(12.dp))
                IconButton(onClick = vm::toggleTheme) {
                    Icon(if (state.darkTheme) Icons.Outlined.LightMode else Icons.Outlined.DarkMode, "Toggle theme")
                }
            }
        }
        Box(Modifier.weight(1f)) {
            when (val load = state.load) {
                LoadState.Loading -> Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(Modifier.height(12.dp))
                    Text("Loading DistilBERT (int8) and TF-IDF model…")
                }
                is LoadState.Failed -> Text("Failed to load models: ${load.message}", Modifier.align(Alignment.Center).padding(24.dp), color = FakeRed)
                LoadState.Ready -> if (tab == 0) AnalyzeScreen(vm, state) else ReportScreen(state.metrics)
            }
        }
        StatusBar(state)
    }
}

@Composable
private fun StatusBar(state: UiState) {
    Surface(tonalElevation = 3.dp) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.Memory, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.secondary)
            Spacer(Modifier.width(6.dp))
            val threads = state.result?.threads ?: minOf(4, Runtime.getRuntime().availableProcessors())
            val cores = Runtime.getRuntime().availableProcessors()
            Text("CPU only · ONNX Runtime · $threads threads of $cores logical cores · no GPU used", style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.weight(1f))
            val status = when (state.load) {
                LoadState.Loading -> "Loading models…"
                is LoadState.Failed -> "Model load failed"
                LoadState.Ready -> state.result?.let { "Last analysis: %.0f ms total".format(it.stages.sumOf { s -> s.millis }) } ?: "Models ready"
            }
            Text(status, style = MaterialTheme.typography.labelSmall)
        }
    }
}
