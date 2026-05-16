pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

// Single-module layout: the Android module lives at this Gradle root.
// No `include(":app")` because there is no `app/` subdirectory; the
// build script `build.gradle.kts` next to this file IS the module.
rootProject.name = "paperclip-remote"
