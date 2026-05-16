// JVM unit test for com.paperclip.remote.input.InputMapper.
//
// Run:
//   kotlinc -d out ../../app/src/main/java/com/paperclip/remote/input/InputMapper.kt
//   kotlinc -cp out -d out_test InputMapperCheck.kt
//   java -cp out:out_test:$KOTLIN_HOME/lib/kotlin-stdlib.jar InputMapperCheckKt
//
// Pure-logic class, no Android dependencies — covers tap mapping under
// STRETCH and FIT modes including letterbox / pillarbox / outside-region.
import com.paperclip.remote.input.InputMapper

fun main() {
    var fail = 0
    fun check(name: String, ok: Boolean, ctx: String = "") {
        if (!ok) { println("FAIL $name $ctx"); fail++ } else println("ok   $name")
    }

    // ---- STRETCH: linear mapping ----
    run {
        val m = InputMapper(1080, 2400, InputMapper.Mode.STRETCH)
        check("stretch center",
            m.mapPoint(500f, 1000f, 1000, 2000) == 540 to 1200)
        check("stretch origin",
            m.mapPoint(0f, 0f, 1000, 2000) == 0 to 0)
        check("stretch far corner (clamped)",
            m.mapPoint(1000f, 2000f, 1000, 2000) == 1079 to 2399)
        check("stretch zero view -> null",
            m.mapPoint(50f, 50f, 0, 0) == null)
    }

    // ---- FIT, equal aspect ----
    run {
        val m = InputMapper(1080, 2400, InputMapper.Mode.FIT)
        check("fit same-aspect center",
            m.mapPoint(270f, 600f, 540, 1200) == 540 to 1200)
        check("fit same-aspect origin",
            m.mapPoint(0f, 0f, 540, 1200) == 0 to 0)
    }

    // ---- FIT pillarbox (view wider than peer) ----
    run {
        val m = InputMapper(1080, 2400, InputMapper.Mode.FIT)
        check("fit pillarbox tap in left bar -> null",
            m.mapPoint(100f, 1000f, 1800, 2000) == null)
        check("fit pillarbox tap on left edge",
            m.mapPoint(450f, 0f, 1800, 2000) == 0 to 0)
        check("fit pillarbox tap on right edge",
            m.mapPoint(450f + 900 - 1, 2000f - 1, 1800, 2000)
              ?.let { it.first in 1077..1079 && it.second in 2397..2399 } == true)
        check("fit pillarbox tap in right bar -> null",
            m.mapPoint(1700f, 1000f, 1800, 2000) == null)
    }

    // ---- FIT letterbox (view taller than peer) ----
    run {
        val m = InputMapper(1920, 1080, InputMapper.Mode.FIT)
        check("fit letterbox tap in top bar -> null",
            m.mapPoint(500f, 100f, 1000, 1000) == null)
        check("fit letterbox tap on top edge",
            m.mapPoint(500f, 219f, 1000, 1000) ==
                (500f / 1000f * 1920).toInt() to 0)
        check("fit letterbox tap in bottom bar -> null",
            m.mapPoint(500f, 900f, 1000, 1000) == null)
    }

    if (fail > 0) { println("\n$fail failures"); kotlin.system.exitProcess(1) }
    println("\nALL OK")
}
