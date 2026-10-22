package fakenews.app

import fakenews.core.Pipeline
import fakenews.core.TextCleaner
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class ModelMetrics(
    val name: String, val accuracy: Double, val precision_fake: Double, val recall_fake: Double,
    val precision_real: Double, val recall_real: Double, val f1_macro: Double,
    val confusion: List<List<Int>>, val size_mb: Double, val latency_ms: Double,
)

@Serializable
data class Metrics(val dataset: String, val test_size: Int, val train_note: String, val latency_note: String, val models: List<ModelMetrics>)

@Serializable
data class Samples(val real: List<String>, val fake: List<String>)

sealed interface LoadState {
    data object Loading : LoadState
    data object Ready : LoadState
    data class Failed(val message: String) : LoadState
}

data class UiState(
    val load: LoadState = LoadState.Loading,
    val input: String = "",
    val tokenCount: Int = 0,
    val truncated: Boolean = false,
    val analyzing: Boolean = false,
    val result: Pipeline.AnalysisResult? = null,
    val metrics: Metrics? = null,
    val samples: Samples = Samples(emptyList(), emptyList()),
    val darkTheme: Boolean = false,
)

class AppViewModel {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()
    private var pipeline: Pipeline? = null
    private val sampleIndex = mutableMapOf("real" to 0, "fake" to 0)
    private val json = Json { ignoreUnknownKeys = true }

    init {
        scope.launch {
            try {
                val (p, metrics, samples) = withContext(Dispatchers.IO) {
                    Triple(
                        Pipeline.load(resourceText("vocab.txt").lines(), resourceBytes("model_int8.onnx"), resourceText("tfidf_lr.json")),
                        runCatching { json.decodeFromString<Metrics>(resourceText("metrics.json")) }.getOrNull(),
                        runCatching { json.decodeFromString<Samples>(resourceText("samples.json")) }.getOrDefault(Samples(emptyList(), emptyList())),
                    )
                }
                pipeline = p
                _state.update { it.copy(load = LoadState.Ready, metrics = metrics, samples = samples) }
                onInput(_state.value.input)
            } catch (e: Throwable) {
                _state.update { it.copy(load = LoadState.Failed(e.message ?: e.toString())) }
            }
        }
    }

    fun onInput(text: String) {
        val enc = pipeline?.tokenizer?.encode(TextCleaner.clean(text))
        _state.update { it.copy(input = text, tokenCount = enc?.fullLength ?: 0, truncated = enc?.truncated ?: false) }
    }

    fun loadSample(kind: String) {
        val list = if (kind == "real") _state.value.samples.real else _state.value.samples.fake
        if (list.isEmpty()) return
        val i = sampleIndex.getValue(kind)
        sampleIndex[kind] = (i + 1) % list.size
        onInput(list[i])
        analyze()
    }

    fun clear() {
        onInput("")
        _state.update { it.copy(result = null) }
    }

    fun toggleTheme() = _state.update { it.copy(darkTheme = !it.darkTheme) }

    fun analyze() {
        val p = pipeline ?: return
        val text = _state.value.input
        if (text.isBlank() || _state.value.analyzing) return
        _state.update { it.copy(analyzing = true) }
        scope.launch {
            val r = withContext(Dispatchers.Default) { p.analyze(text) }
            _state.update { it.copy(analyzing = false, result = r) }
        }
    }

    fun close() = pipeline?.close()

    private fun resourceBytes(name: String): ByteArray =
        (AppViewModel::class.java.classLoader.getResourceAsStream(name) ?: error("Missing resource $name. Run the training scripts first."))
            .use { it.readBytes() }

    private fun resourceText(name: String) = resourceBytes(name).toString(Charsets.UTF_8)
}
