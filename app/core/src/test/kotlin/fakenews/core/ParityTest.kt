package fakenews.core

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Kotlin must reproduce the Python pipeline exactly (fixtures from training/06_make_fixtures.py). */
class ParityTest {
    @Serializable
    data class Fixture(val raw: String, val cleaned: String, val input_ids: List<Long>, val logits: List<Float>, val logits_fp32: List<Float>, val tfidf_prob_real: Double)

    private val dir = File(System.getProperty("artifacts.dir"))
    private val fixtures = Json.decodeFromString<List<Fixture>>(File(dir, "fixtures.json").readText())
    private val tokenizer = WordPieceTokenizer(File(dir, "vocab.txt").readLines())

    @Test
    fun cleanerMatchesPython() = fixtures.forEach { assertEquals(it.cleaned, TextCleaner.clean(it.raw), "raw=${it.raw.take(60)}") }

    @Test
    fun tokenizerMatchesHuggingFace() = fixtures.forEach {
        assertContentEquals(it.input_ids.toLongArray(), tokenizer.encode(it.cleaned).ids, "text=${it.cleaned.take(60)}")
    }

    @Test
    fun tfidfMatchesSklearn() {
        val clf = TfidfLrClassifier(File(dir, "tfidf_lr.json").readText())
        fixtures.forEach { assertTrue(abs(it.tfidf_prob_real - clf.predict(it.cleaned).probReal) < 1e-4, "text=${it.cleaned.take(60)}") }
    }

    /** fp32 kernels are deterministic across ONNX Runtime versions, so logits must match closely. */
    @Test
    fun distilbertFp32MatchesPython() {
        val fp32 = File(dir, "../models/distilbert-fakenews/model_fp32.onnx")
        if (!fp32.exists()) return println("skipped: $fp32 not found (only present on the training machine)")
        DistilBertClassifier(fp32.readBytes()).use { bert ->
            var worst = 0f
            fixtures.forEach { f ->
                val r = bert.predict(f.input_ids.toLongArray())
                f.logits_fp32.forEachIndexed { i, v -> worst = maxOf(worst, abs(v - r.logits[i])) }
            }
            println("fp32: max |logit diff| vs Python = $worst")
            assertTrue(worst < 1e-3)
        }
    }

    /** int8 kernels may round slightly differently between ORT versions: require the same predictions. */
    @Test
    fun distilbertInt8AgreesWithPython() {
        @Serializable data class Row(val text: String, val label: Int, val pred: Int)
        val rows = Json.decodeFromString<List<Row>>(File(dir, "../data/agreement.json").readText())
        DistilBertClassifier(File(dir, "model_int8.onnx").readBytes()).use { bert ->
            var worst = 0f
            fixtures.forEach { f ->
                val r = bert.predict(f.input_ids.toLongArray())
                f.logits.forEachIndexed { i, v -> worst = maxOf(worst, abs(v - r.logits[i])) }
            }
            val preds = rows.map { if (bert.predict(tokenizer.encode(it.text).ids).probReal >= 0.5) 1 else 0 }
            val agreement = rows.indices.count { preds[it] == rows[it].pred }.toDouble() / rows.size
            val accuracy = rows.indices.count { preds[it] == rows[it].label }.toDouble() / rows.size
            println("int8: max |logit diff| = $worst, agreement with Python = ${"%.4f".format(agreement)}, accuracy = ${"%.4f".format(accuracy)} on ${rows.size}")
            assertTrue(agreement >= 0.99)
        }
    }
}
