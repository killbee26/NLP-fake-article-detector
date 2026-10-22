plugins {
    kotlin("jvm")
    kotlin("plugin.serialization")
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    // 1.17.x: built before MSVC 14.40, so it works with the older msvcp140.dll that many JDKs (and the
    // jpackage-bundled runtime) ship. Newer ORT builds crash with "DLL initialization routine failed" there.
    api("com.microsoft.onnxruntime:onnxruntime:1.17.3")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
    // Parity tests read the Python-generated artifacts directly.
    systemProperty("artifacts.dir", rootProject.file("../artifacts").absolutePath)
    testLogging { events("passed", "failed"); showStandardStreams = true }
}
