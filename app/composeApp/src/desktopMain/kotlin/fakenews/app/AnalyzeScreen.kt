package fakenews.app

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.compositeOver
import androidx.compose.ui.input.key.*
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import fakenews.core.Pipeline
import kotlin.math.abs

@Composable
fun AnalyzeScreen(vm: AppViewModel, state: UiState) {
    Row(Modifier.fillMaxSize().padding(16.dp), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        InputPanel(vm, state, Modifier.weight(0.42f).fillMaxHeight())
        Column(Modifier.weight(0.58f).fillMaxHeight().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            val r = state.result
            if (r == null) EmptyHint() else {
                VerdictCard(r)
                ComparisonRow(r)
                InfluenceCard(r)
                PipelineCard(r)
            }
        }
    }
}

@Composable
private fun InputPanel(vm: AppViewModel, state: UiState, modifier: Modifier) {
    Card(modifier) {
        Column(Modifier.fillMaxSize().padding(16.dp)) {
            Text("News article", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = state.input, onValueChange = vm::onInput,
                modifier = Modifier.fillMaxWidth().weight(1f).onPreviewKeyEvent {
                    if (it.type == KeyEventType.KeyDown && it.key == Key.Enter && it.isCtrlPressed) { vm.analyze(); true } else false
                },
                placeholder = { Text("Paste a headline + article text here, or load a sample…") },
                textStyle = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(6.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                val words = state.input.split(Regex("\\s+")).count { it.isNotBlank() }
                Text("$words words · ${state.tokenCount} WordPiece tokens", style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.secondary)
                if (state.truncated) {
                    Spacer(Modifier.width(8.dp))
                    Icon(Icons.Outlined.WarningAmber, null, Modifier.size(14.dp), tint = UnsureAmber)
                    Text(" DistilBERT reads only the first 128 tokens", style = MaterialTheme.typography.labelMedium, color = UnsureAmber)
                }
            }
            Spacer(Modifier.height(10.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(onClick = { vm.loadSample("real") }) { Text("Sample: Real") }
                OutlinedButton(onClick = { vm.loadSample("fake") }) { Text("Sample: Fake") }
                TextButton(onClick = vm::clear) { Text("Clear") }
                Spacer(Modifier.weight(1f))
                Button(onClick = vm::analyze, enabled = state.input.isNotBlank() && !state.analyzing) {
                    if (state.analyzing) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp, color = MaterialTheme.colorScheme.onPrimary)
                    else Icon(Icons.Outlined.Search, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Analyze")
                }
            }
            Text("Tip: Ctrl+Enter to analyze", style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.secondary, modifier = Modifier.align(Alignment.End).padding(top = 4.dp))
        }
    }
}

@Composable
private fun EmptyHint() {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(28.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(Icons.Outlined.Newspaper, null, Modifier.size(48.dp), tint = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(12.dp))
            Text("Paste an article or load a sample, then press Analyze.", style = MaterialTheme.typography.titleSmall)
            Spacer(Modifier.height(6.dp))
            Text("You will see the verdict, both models side by side, which words drove the decision, and every step of the NLP pipeline.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
        }
    }
}

@Composable
private fun VerdictCard(r: Pipeline.AnalysisResult) {
    val p = r.bert.probReal
    val color = verdictColor(p)
    val conf = maxOf(p, 1 - p)
    val animated by animateFloatAsState(conf.toFloat())
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
        containerColor = color.copy(alpha = 0.12f).compositeOver(MaterialTheme.colorScheme.surface),
        contentColor = MaterialTheme.colorScheme.onSurface,
    )) {
        Row(Modifier.padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Verdict (DistilBERT)", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.secondary)
                Text(if (conf < 0.65) "UNCERTAIN · leans ${verdictLabel(p)}" else verdictLabel(p),
                    fontSize = 40.sp, fontWeight = FontWeight.ExtraBold, color = color)
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(progress = { animated }, Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(5.dp)),
                    color = color, trackColor = color.copy(alpha = 0.2f))
                Spacer(Modifier.height(6.dp))
                Text("Confidence %.1f%%   ·   P(fake) %.3f   ·   P(real) %.3f".format(conf * 100, 1 - p, p),
                    style = MaterialTheme.typography.bodySmall)
            }
            Spacer(Modifier.width(16.dp))
            Icon(if (p >= 0.5) Icons.Outlined.VerifiedUser else Icons.Outlined.GppBad, null, Modifier.size(64.dp), tint = color)
        }
    }
}

@Composable
private fun ComparisonRow(r: Pipeline.AnalysisResult) {
    val agree = (r.bert.probReal >= 0.5) == (r.tfidf.probReal >= 0.5)
    Column {
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            ModelCard("DistilBERT · int8 ONNX", "Transformer, transfer learning", r.bert.probReal, r.bertMillis, Modifier.weight(1f))
            ModelCard("TF-IDF + Logistic Regression", "1-2 gram baseline · ${r.tfidf.matchedTerms} features hit", r.tfidf.probReal, r.tfidfMillis, Modifier.weight(1f))
        }
        Spacer(Modifier.height(6.dp))
        AssistChip(onClick = {}, label = { Text(if (agree) "Both models agree" else "Models disagree: the transformer and the baseline see this differently") },
            leadingIcon = { Icon(if (agree) Icons.Outlined.CheckCircle else Icons.Outlined.CompareArrows, null, Modifier.size(16.dp)) })
    }
}

@Composable
private fun ModelCard(title: String, subtitle: String, probReal: Double, ms: Double, modifier: Modifier) {
    val color = verdictColor(probReal)
    Card(modifier) {
        Column(Modifier.padding(14.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Text(subtitle, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.secondary)
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.Bottom) {
                Text(verdictLabel(probReal), fontSize = 22.sp, fontWeight = FontWeight.Bold, color = color)
                Spacer(Modifier.width(8.dp))
                Text("%.1f%%".format(maxOf(probReal, 1 - probReal) * 100), style = MaterialTheme.typography.bodyMedium)
                Spacer(Modifier.weight(1f))
                Text("%.1f ms".format(ms), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.secondary)
            }
            Spacer(Modifier.height(6.dp))
            // Bar from FAKE (left) to REAL (right)
            Box(Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)).background(FakeRed.copy(alpha = 0.25f))) {
                Box(Modifier.fillMaxWidth(probReal.toFloat()).fillMaxHeight().background(RealGreen.copy(alpha = 0.8f)))
            }
            Row {
                Text("fake", style = MaterialTheme.typography.labelSmall, color = FakeRed)
                Spacer(Modifier.weight(1f))
                Text("real", style = MaterialTheme.typography.labelSmall, color = RealGreen)
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun InfluenceCard(r: Pipeline.AnalysisResult) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("Why? Word influence (TF-IDF + LR weights)", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Text("Red words push towards FAKE, green towards REAL. Darker = stronger.", style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.secondary)
            Spacer(Modifier.height(10.dp))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                r.tfidf.topTerms.forEach { t ->
                    val c = if (t.contribution > 0) RealGreen else FakeRed
                    Surface(shape = RoundedCornerShape(12.dp), color = c.copy(alpha = 0.15f)) {
                        Text("${t.term}  ${"%+.2f".format(t.contribution)}", Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelMedium, color = c)
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            val maxW = r.wordWeights.values.maxOfOrNull { abs(it) }?.takeIf { it > 0 } ?: 1.0
            val text = buildAnnotatedString {
                val words = Regex("[A-Za-z0-9]+|[^A-Za-z0-9]+").findAll(r.cleaned.take(2500)).map { it.value }
                for (w in words) {
                    val weight = r.wordWeights[w.lowercase()]
                    if (weight == null || abs(weight) / maxW < 0.08) append(w)
                    else withStyle(SpanStyle(background = (if (weight > 0) RealGreen else FakeRed).copy(alpha = (0.15 + 0.6 * abs(weight) / maxW).toFloat().coerceAtMost(0.75f)))) { append(w) }
                }
                if (r.cleaned.length > 2500) append(" …")
            }
            SelectionContainer { Text(text, style = MaterialTheme.typography.bodyMedium, lineHeight = 22.sp) }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PipelineCard(r: Pipeline.AnalysisResult) {
    var expanded by remember { mutableStateOf(true) }
    val ms = r.stages.associate { it.name to it.millis }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Row(Modifier.fillMaxWidth().clickable { expanded = !expanded }, verticalAlignment = Alignment.CenterVertically) {
                Text("NLP pipeline, step by step", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                Icon(if (expanded) Icons.Outlined.ExpandLess else Icons.Outlined.ExpandMore, null)
            }
            AnimatedVisibility(expanded) {
                Column(Modifier.padding(top = 8.dp)) {
                    Step(1, "Raw input", "${r.raw.length} characters", null) {
                        Mono(r.raw.take(220) + if (r.raw.length > 220) " …" else "")
                    }
                    Step(2, "Preprocessing", "removed datelines, URLs, @handles, image credits; normalized whitespace · ${r.cleaned.length} chars", ms["Clean"]) {
                        Mono(r.cleaned.take(220) + if (r.cleaned.length > 220) " …" else "")
                    }
                    Step(3, "WordPiece tokenization", "${r.encoding.fullLength} tokens" + if (r.encoding.truncated) " → truncated to 128" else "", ms["WordPiece"]) {
                        FlowRow(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            r.encoding.tokens.take(60).forEach { tok ->
                                val special = tok.startsWith("[")
                                val sub = tok.startsWith("##")
                                val bg = when {
                                    special -> MaterialTheme.colorScheme.primary.copy(alpha = 0.25f)
                                    sub -> UnsureAmber.copy(alpha = 0.25f)
                                    else -> MaterialTheme.colorScheme.surfaceVariant
                                }
                                Text(tok, Modifier.clip(RoundedCornerShape(4.dp)).background(bg).padding(horizontal = 5.dp, vertical = 2.dp),
                                    fontFamily = FontFamily.Monospace, fontSize = 12.sp)
                            }
                            if (r.encoding.tokens.size > 60) Text("… +${r.encoding.tokens.size - 60} more", fontSize = 12.sp)
                        }
                    }
                    Step(4, "Token IDs (model input)", "vocabulary of 30,522 · attention mask all 1s", null) {
                        Mono(r.encoding.ids.take(40).joinToString(", ", "[", if (r.encoding.ids.size > 40) ", …]" else "]"))
                    }
                    Step(5, "DistilBERT forward pass → logits", "6 transformer layers, int8 weights, ONNX Runtime CPU", ms["DistilBERT"]) {
                        Mono("logits = [fake: %.4f, real: %.4f]".format(r.bert.logits[0], r.bert.logits[1]))
                    }
                    Step(6, "Softmax → probabilities", "", null, last = true) {
                        Mono("P(fake) = %.4f   P(real) = %.4f   → %s".format(r.bert.probs[0], r.bert.probs[1], verdictLabel(r.bert.probReal)))
                    }
                }
            }
        }
    }
}

@Composable
private fun Step(n: Int, title: String, detail: String, millis: Double?, last: Boolean = false, content: @Composable () -> Unit) {
    Row(Modifier.fillMaxWidth().height(IntrinsicSize.Min)) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Box(Modifier.size(24.dp).clip(RoundedCornerShape(12.dp)).background(MaterialTheme.colorScheme.primary), contentAlignment = Alignment.Center) {
                Text("$n", color = MaterialTheme.colorScheme.onPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold)
            }
            if (!last) Box(Modifier.width(2.dp).weight(1f).background(MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)))
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f).padding(bottom = 14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(title, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                if (millis != null) Text("%.2f ms".format(millis), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.secondary)
            }
            if (detail.isNotEmpty()) Text(detail, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.secondary)
            Spacer(Modifier.height(4.dp))
            content()
        }
    }
}

@Composable
private fun Mono(text: String) = Text(
    text, Modifier.fillMaxWidth().clip(RoundedCornerShape(6.dp)).background(MaterialTheme.colorScheme.surfaceVariant).padding(8.dp),
    fontFamily = FontFamily.Monospace, fontSize = 12.sp, color = Color.Unspecified,
)
