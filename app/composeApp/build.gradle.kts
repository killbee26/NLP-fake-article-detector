import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    kotlin("multiplatform")
    kotlin("plugin.compose")
    kotlin("plugin.serialization")
    id("org.jetbrains.compose")
}

kotlin {
    jvmToolchain(21)
    jvm("desktop")

    sourceSets {
        val desktopMain by getting {
            // Bundle the trained model, vocab, baseline and metrics into the app jar.
            resources.srcDir(rootProject.file("../artifacts"))
            dependencies {
                implementation(project(":core"))
                implementation(compose.desktop.currentOs)
                implementation(compose.material3)
                implementation(compose.materialIconsExtended)
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.10.2")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")
            }
        }
    }
}

tasks.register<JavaExec>("screenshots") {
    group = "application"
    description = "Render the UI headlessly to PNGs in /screenshots"
    val jar = tasks.named("desktopJar")
    dependsOn(jar)
    classpath(jar, configurations.named("desktopRuntimeClasspath"))
    mainClass.set("fakenews.app.ScreenshotsKt")
    args(rootProject.file("../screenshots").absolutePath)
    systemProperty("java.awt.headless", "true")
}

compose.desktop {
    application {
        mainClass = "fakenews.app.MainKt"
        jvmArgs += listOf("-Xmx1g")
        nativeDistributions {
            targetFormats(TargetFormat.Msi, TargetFormat.Exe, TargetFormat.Deb, TargetFormat.Dmg)
            packageName = "FakeNewsDetector"
            packageVersion = "1.0.0"
        }
    }
}
