package fakenews.core

/**
 * Mirrors `clean_text()` in training/common.py so the app sees text exactly as the models did in training.
 * (?U) makes \w, \s and \b Unicode-aware like Python's `re`.
 */
object TextCleaner {
    private val dateline = Regex("""(?iU)^.{0,120}?\((reuters|ap)\)\s*-\s*""")
    private val reuters = Regex("""(?iU)\breuters\b""")
    private val url = Regex("""(?iU)(https?://\S+|www\.\S+|pic\.twitter\.com/\S+)""")
    private val handle = Regex("""(?U)@\w+""")
    private val credit = Regex("""(?iU)(featured image|photo by|image via|getty images|21st century wire)[^.]*\.?""")
    private val whitespace = Regex("""(?U)\s+""")

    fun clean(text: String): String {
        var t = dateline.replaceFirst(text, "")
        t = reuters.replace(t, "")
        t = url.replace(t, " ")
        t = handle.replace(t, " ")
        t = credit.replace(t, " ")
        return whitespace.replace(t, " ").trim()
    }
}
