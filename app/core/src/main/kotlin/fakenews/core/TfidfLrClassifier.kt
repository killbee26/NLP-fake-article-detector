package fakenews.core

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * TF-IDF (1-2 grams, sublinear tf, L2 norm) + Logistic Regression, reimplemented from the
 * sklearn weights exported by training/02_baseline_tfidf.py. Positive score -> REAL.
 */
class TfidfLrClassifier(json: String) {
    @Serializable
    private data class Weights(
        val ngram_max: Int, val sublinear_tf: Boolean, val vocab: Map<String, Int>,
        val idf: List<Double>, val coef: List<Double>, val intercept: Double,
    )

    data class Term(val term: String, val contribution: Double)
    data class Result(val probReal: Double, val topTerms: List<Term>, val matchedTerms: Int)

    private val w = Json { ignoreUnknownKeys = true }.decodeFromString<Weights>(json)
    private val termOf: Map<Int, String> = w.vocab.entries.associate { it.value to it.key }
    private val token = Regex("[a-z0-9]+")

    fun tokenize(text: String): List<String> = token.findAll(text.lowercase()).map { it.value }.toList()

    fun predict(text: String): Result {
        val toks = tokenize(text)
        val counts = HashMap<Int, Int>()
        for (n in 1..w.ngram_max) {
            for (i in 0..toks.size - n) {
                val idx = w.vocab[toks.subList(i, i + n).joinToString(" ")] ?: continue
                counts.merge(idx, 1, Int::plus)
            }
        }
        val values = counts.mapValues { (idx, c) -> (if (w.sublinear_tf) 1 + ln(c.toDouble()) else c.toDouble()) * w.idf[idx] }
        val norm = sqrt(values.values.sumOf { it * it }).takeIf { it > 0 } ?: 1.0
        val contributions = values.mapValues { (idx, v) -> w.coef[idx] * v / norm }
        val score = contributions.values.sum() + w.intercept
        val top = contributions.mapKeys { termOf.getValue(it.key) }.entries.sortedByDescending { kotlin.math.abs(it.value) }.take(12).map { Term(it.key, it.value) }
        return Result(1.0 / (1.0 + exp(-score)), top, counts.size)
    }

    /** Influence of each unigram (for highlighting words in the UI). */
    fun unigramWeights(text: String): Map<String, Double> =
        tokenize(text).distinct().mapNotNull { t -> w.vocab[t]?.let { t to w.coef[it] } }.toMap()
}
