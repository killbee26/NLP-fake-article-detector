package fakenews.core

/** End-to-end analysis: clean -> tokenize -> both classifiers, with per-stage timings. */
class Pipeline(
    val tokenizer: WordPieceTokenizer,
    private val bert: DistilBertClassifier,
    private val tfidf: TfidfLrClassifier,
) : AutoCloseable {

    data class Stage(val name: String, val millis: Double)

    data class AnalysisResult(
        val raw: String,
        val cleaned: String,
        val encoding: WordPieceTokenizer.Encoding,
        val bert: DistilBertClassifier.Result,
        val tfidf: TfidfLrClassifier.Result,
        val wordWeights: Map<String, Double>,
        val stages: List<Stage>,
        val threads: Int,
    ) {
        val bertMillis get() = stages.filter { it.name != "TF-IDF + LR" }.sumOf { it.millis }
        val tfidfMillis get() = stages.first { it.name == "TF-IDF + LR" }.millis + stages.first().millis
    }

    fun analyze(text: String): AnalysisResult {
        val stages = mutableListOf<Stage>()
        fun <T> timed(name: String, block: () -> T): T {
            val t0 = System.nanoTime()
            return block().also { stages += Stage(name, (System.nanoTime() - t0) / 1e6) }
        }
        val cleaned = timed("Clean") { TextCleaner.clean(text) }
        val enc = timed("WordPiece") { tokenizer.encode(cleaned) }
        val bertRes = timed("DistilBERT") { bert.predict(enc.ids) }
        val tfidfRes = timed("TF-IDF + LR") { tfidf.predict(cleaned) }
        return AnalysisResult(text, cleaned, enc, bertRes, tfidfRes, tfidf.unigramWeights(cleaned), stages, bert.threads)
    }

    override fun close() = bert.close()

    companion object {
        fun load(vocab: List<String>, modelBytes: ByteArray, tfidfJson: String) =
            Pipeline(WordPieceTokenizer(vocab), DistilBertClassifier(modelBytes), TfidfLrClassifier(tfidfJson))
    }
}
