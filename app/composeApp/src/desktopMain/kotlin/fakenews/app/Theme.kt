package fakenews.app

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val FakeRed = Color(0xFFD64545)
val RealGreen = Color(0xFF2E9E5B)
val UnsureAmber = Color(0xFFE0A030)

private val Light = lightColorScheme(
    primary = Color(0xFF3B5BDB), secondary = Color(0xFF5C677D),
    background = Color(0xFFF3F4F8), surface = Color.White, surfaceVariant = Color(0xFFEDEFF5),
    surfaceContainerHighest = Color.White, surfaceContainerHigh = Color.White, surfaceContainer = Color(0xFFF8F9FC),
    surfaceContainerLow = Color(0xFFF8F9FC), surfaceContainerLowest = Color.White,
)
private val Dark = darkColorScheme(
    primary = Color(0xFF8EA4FF), secondary = Color(0xFFAAB4C8),
    background = Color(0xFF121418), surface = Color(0xFF1B1E24), surfaceVariant = Color(0xFF262A33),
    surfaceContainerHighest = Color(0xFF1E2128), surfaceContainerHigh = Color(0xFF1E2128), surfaceContainer = Color(0xFF1B1E24),
    surfaceContainerLow = Color(0xFF1B1E24), surfaceContainerLowest = Color(0xFF15171C),
)

@Composable
fun AppTheme(dark: Boolean, content: @Composable () -> Unit) =
    MaterialTheme(colorScheme = if (dark) Dark else Light, content = content)

/** Color for a verdict: red = fake, green = real, amber when the model is unsure. */
fun verdictColor(probReal: Double): Color = when {
    maxOf(probReal, 1 - probReal) < 0.65 -> UnsureAmber
    probReal >= 0.5 -> RealGreen
    else -> FakeRed
}

fun verdictLabel(probReal: Double) = if (probReal >= 0.5) "REAL" else "FAKE"
