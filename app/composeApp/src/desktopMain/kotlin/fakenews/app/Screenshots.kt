package fakenews.app

import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.unit.Density
import androidx.compose.ui.use
import fakenews.core.Pipeline
import kotlinx.serialization.json.Json
import org.jetbrains.skia.EncodedImageFormat
import java.io.File

/** Renders the UI headlessly to PNGs (for the report and for checking the layout). Run: gradlew :composeApp:screenshots */
fun main(args: Array<String>) {
    val out = File(args.firstOrNull() ?: "screenshots").apply { mkdirs() }
    val res = { name: String -> Pipeline::class.java.classLoader.getResourceAsStream(name)!!.use { it.readBytes() } }
    val json = Json { ignoreUnknownKeys = true }
    val pipeline = Pipeline.load(res("vocab.txt").toString(Charsets.UTF_8).lines(), res("model_int8.onnx"), res("tfidf_lr.json").toString(Charsets.UTF_8))
    val samples = json.decodeFromString<Samples>(res("samples.json").toString(Charsets.UTF_8))
    val metrics = json.decodeFromString<Metrics>(res("metrics.json").toString(Charsets.UTF_8))
    val vm = AppViewModel()

    fun shot(name: String, state: UiState, tab: Int, height: Int = 1100) {
        ImageComposeScene(1400, height, Density(1f)) { AppTheme(state.darkTheme) { App(vm, state, initialTab = tab) } }.use { scene ->
            scene.render() // first frame
            val img = scene.render(500_000_000L)
            File(out, "$name.png").writeBytes(img.encodeToData(EncodedImageFormat.PNG)!!.bytes)
            println("wrote ${File(out, "$name.png").absolutePath}")
        }
    }

    fun analyzed(text: String, dark: Boolean = false): UiState {
        val r = pipeline.analyze(text)
        val enc = pipeline.tokenizer.encode(r.cleaned)
        return UiState(LoadState.Ready, text, enc.fullLength, enc.truncated, false, r, metrics, samples, dark)
    }

    shot("analyze_fake", analyzed(samples.fake.first()), 0, 1500)
    shot("analyze_real", analyzed(samples.real.first()), 0, 1500)
    shot("analyze_dark", analyzed(samples.fake[1], dark = true), 0, 1500)
    shot("report", UiState(LoadState.Ready, metrics = metrics), 1, 820)
    pipeline.close()
    vm.close()
    kotlin.system.exitProcess(0)
}
