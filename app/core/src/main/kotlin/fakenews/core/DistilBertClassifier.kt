package fakenews.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlin.math.exp

/** Runs the int8-quantized DistilBERT ONNX model on the CPU via ONNX Runtime. */
class DistilBertClassifier(modelBytes: ByteArray, val threads: Int = minOf(4, Runtime.getRuntime().availableProcessors())) :
    AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelBytes, OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    })

    data class Result(val logits: FloatArray, val probs: DoubleArray) {
        val probReal get() = probs[1]
    }

    fun predict(ids: LongArray): Result {
        val shape = longArrayOf(1, ids.size.toLong())
        val mask = LongArray(ids.size) { 1L }
        OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(ids), shape).use { idsT ->
            OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(mask), shape).use { maskT ->
                session.run(mapOf("input_ids" to idsT, "attention_mask" to maskT)).use { out ->
                    @Suppress("UNCHECKED_CAST")
                    val logits = (out[0].value as Array<FloatArray>)[0]
                    return Result(logits, softmax(logits))
                }
            }
        }
    }

    private fun softmax(x: FloatArray): DoubleArray {
        val m = x.max()
        val e = x.map { exp((it - m).toDouble()) }
        val s = e.sum()
        return e.map { it / s }.toDoubleArray()
    }

    override fun close() = session.close()

    companion object {
        init { preloadSystemMsvcRuntime() }

        /**
         * Some JDKs (e.g. Android Studio's JBR) ship an old msvcp140.dll in their bin folder, which Windows picks up
         * first. Recent ONNX Runtime builds then fail with "DLL initialization routine failed". Loading the
         * (newer) system copy first makes ONNX Runtime bind to it instead.
         */
        private fun preloadSystemMsvcRuntime() {
            if (!System.getProperty("os.name").startsWith("Windows")) return
            val sys32 = java.io.File(System.getenv("SystemRoot") ?: "C:\\Windows", "System32")
            for (dll in listOf("msvcp140.dll", "msvcp140_1.dll")) {
                val f = java.io.File(sys32, dll)
                if (f.exists()) runCatching { System.load(f.absolutePath) }
            }
        }
    }
}
