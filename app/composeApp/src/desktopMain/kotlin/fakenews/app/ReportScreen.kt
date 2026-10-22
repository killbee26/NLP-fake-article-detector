package fakenews.app

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

@Composable
fun ReportScreen(metrics: Metrics?) {
    if (metrics == null) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("metrics.json not found. Run training/05_evaluate.py.") }
        return
    }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) {
                Text("Evaluation on held-out test set (${metrics.test_size} articles)", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(4.dp))
                Text("Dataset: ${metrics.dataset}", style = MaterialTheme.typography.bodySmall)
                Text("Training: ${metrics.train_note}", style = MaterialTheme.typography.bodySmall)
                Text("Latency: ${metrics.latency_note}", style = MaterialTheme.typography.bodySmall)
                Spacer(Modifier.height(14.dp))
                MetricsTable(metrics.models)
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            metrics.models.forEach { m -> ConfusionCard(m, Modifier.weight(1f)) }
        }
    }
}

@Composable
private fun MetricsTable(models: List<ModelMetrics>) {
    val headers = listOf("Model", "Accuracy", "Macro F1", "Prec. fake", "Recall fake", "Prec. real", "Recall real", "Size (MB)", "CPU latency (ms)")
    val best = models.maxOf { it.f1_macro }
    Column {
        Row(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant).padding(vertical = 8.dp)) {
            headers.forEachIndexed { i, h -> Cell(h, if (i == 0) 2.2f else 1f, bold = true) }
        }
        models.forEach { m ->
            HorizontalDivider()
            Row(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
                Cell(m.name + if (m.f1_macro == best) "  ★" else "", 2.2f, bold = m.f1_macro == best)
                listOf(m.accuracy, m.f1_macro, m.precision_fake, m.recall_fake, m.precision_real, m.recall_real)
                    .forEach { Cell("%.2f%%".format(it * 100), 1f) }
                Cell("%.1f".format(m.size_mb), 1f)
                Cell("%.1f".format(m.latency_ms), 1f)
            }
        }
    }
}

@Composable
private fun RowScope.Cell(text: String, weight: Float, bold: Boolean = false) = Text(
    text, Modifier.weight(weight).padding(horizontal = 6.dp), style = MaterialTheme.typography.bodySmall,
    fontWeight = if (bold) FontWeight.SemiBold else FontWeight.Normal,
)

@Composable
private fun ConfusionCard(m: ModelMetrics, modifier: Modifier) {
    val labels = listOf("FAKE", "REAL")
    val total = m.confusion.flatten().sum().coerceAtLeast(1)
    Card(modifier) {
        Column(Modifier.padding(16.dp)) {
            Text(m.name, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Text("Confusion matrix (rows = true, columns = predicted)", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.secondary)
            Spacer(Modifier.height(10.dp))
            Row { Spacer(Modifier.width(52.dp)); labels.forEach { Text(it, Modifier.weight(1f), textAlign = TextAlign.Center, style = MaterialTheme.typography.labelMedium) } }
            m.confusion.forEachIndexed { i, row ->
                Row(Modifier.padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(labels[i], Modifier.width(52.dp), style = MaterialTheme.typography.labelMedium)
                    row.forEachIndexed { j, v ->
                        val color = if (i == j) RealGreen else FakeRed
                        val alpha = (0.12f + 0.75f * v.toFloat() / total * 2).coerceAtMost(0.85f)
                        Box(Modifier.weight(1f).height(56.dp).padding(horizontal = 3.dp).clip(RoundedCornerShape(8.dp)).background(color.copy(alpha = alpha)),
                            contentAlignment = Alignment.Center) {
                            Text("$v", fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }
    }
}
