package fakenews.core

import java.text.Normalizer

/**
 * Pure-Kotlin re-implementation of HuggingFace's BERT uncased tokenizer
 * (BertNormalizer -> BertPreTokenizer -> WordPiece -> [CLS] ... [SEP]).
 * No native libraries, so it works offline on any JVM.
 */
class WordPieceTokenizer(vocabLines: List<String>, private val maxLength: Int = 128) {
    private val vocab: Map<String, Int> = vocabLines.withIndex()
        .filter { it.value.isNotEmpty() }
        .associate { it.value to it.index }
    private val clsId = vocab.getValue("[CLS]")
    private val sepId = vocab.getValue("[SEP]")
    private val unkId = vocab.getValue("[UNK]")

    data class Encoding(val tokens: List<String>, val ids: LongArray, val truncated: Boolean, val fullLength: Int)

    fun encode(text: String): Encoding {
        val pieces = mutableListOf<String>()
        for (word in preTokenize(normalize(text))) pieces += wordPiece(word)
        val limit = maxLength - 2
        val kept = pieces.take(limit)
        val tokens = listOf("[CLS]") + kept + listOf("[SEP]")
        val ids = LongArray(tokens.size)
        ids[0] = clsId.toLong()
        kept.forEachIndexed { i, p -> ids[i + 1] = (vocab[p] ?: unkId).toLong() }
        ids[ids.size - 1] = sepId.toLong()
        return Encoding(tokens, ids, pieces.size > limit, pieces.size + 2)
    }

    /** Clean control chars, space out CJK, strip accents, lowercase. */
    private fun normalize(text: String): String {
        val sb = StringBuilder()
        text.codePoints().forEach { cp ->
            when {
                cp == 0 || cp == 0xFFFD || isControl(cp) -> Unit
                isWhitespace(cp) -> sb.append(' ')
                isChinese(cp) -> sb.append(' ').appendCodePoint(cp).append(' ')
                else -> sb.appendCodePoint(cp)
            }
        }
        val nfd = Normalizer.normalize(sb, Normalizer.Form.NFD)
        val stripped = StringBuilder()
        nfd.codePoints().forEach { cp ->
            if (Character.getType(cp) != Character.NON_SPACING_MARK.toInt()) stripped.appendCodePoint(cp)
        }
        return stripped.toString().lowercase()
    }

    /** Split on whitespace and isolate every punctuation character. */
    private fun preTokenize(text: String): List<String> {
        val words = mutableListOf<String>()
        val cur = StringBuilder()
        fun flush() { if (cur.isNotEmpty()) { words += cur.toString(); cur.setLength(0) } }
        text.codePoints().forEach { cp ->
            when {
                isWhitespace(cp) -> flush()
                isPunctuation(cp) -> { flush(); words += String(Character.toChars(cp)) }
                else -> cur.appendCodePoint(cp)
            }
        }
        flush()
        return words
    }

    /** Greedy longest-match-first WordPiece; the whole word becomes [UNK] if any part is unknown. */
    private fun wordPiece(word: String): List<String> {
        if (word.codePointCount(0, word.length) > 100) return listOf("[UNK]")
        val out = mutableListOf<String>()
        var start = 0
        while (start < word.length) {
            var end = word.length
            var match: String? = null
            while (start < end) {
                val sub = (if (start > 0) "##" else "") + word.substring(start, end)
                if (sub in vocab) { match = sub; break }
                end = word.offsetByCodePoints(end, -1)
            }
            if (match == null) return listOf("[UNK]")
            out += match
            start = end
        }
        return out
    }

    private fun isWhitespace(cp: Int) =
        cp == '\t'.code || cp == '\n'.code || cp == '\r'.code || Character.isWhitespace(cp) || Character.isSpaceChar(cp)

    private fun isControl(cp: Int): Boolean {
        if (cp == '\t'.code || cp == '\n'.code || cp == '\r'.code) return false
        return when (Character.getType(cp).toByte()) {
            Character.CONTROL, Character.FORMAT, Character.UNASSIGNED, Character.PRIVATE_USE, Character.SURROGATE -> true
            else -> false
        }
    }

    private fun isPunctuation(cp: Int): Boolean {
        if (cp in 33..47 || cp in 58..64 || cp in 91..96 || cp in 123..126) return true
        return when (Character.getType(cp).toByte()) {
            Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION, Character.START_PUNCTUATION,
            Character.END_PUNCTUATION, Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION,
            Character.OTHER_PUNCTUATION -> true
            else -> false
        }
    }

    private fun isChinese(cp: Int) =
        cp in 0x4E00..0x9FFF || cp in 0x3400..0x4DBF || cp in 0x20000..0x2A6DF || cp in 0x2A700..0x2B73F ||
            cp in 0x2B740..0x2B81F || cp in 0x2B820..0x2CEAF || cp in 0xF900..0xFAFF || cp in 0x2F800..0x2FA1F
}
