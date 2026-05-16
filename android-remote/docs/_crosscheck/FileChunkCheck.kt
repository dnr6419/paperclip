// JVM unit test for com.paperclip.remote.files.FileChunk.
//
// Run:
//   kotlinc -d out ../../app/src/main/java/com/paperclip/remote/files/FileChunk.kt
//   kotlinc -cp out -d out_test FileChunkCheck.kt
//   java -cp out:out_test:$KOTLIN_HOME/lib/kotlin-stdlib.jar FileChunkCheckKt
import com.paperclip.remote.files.FileChunk

fun main() {
    var fail = 0
    fun check(name: String, ok: Boolean) {
        if (!ok) { println("FAIL $name"); fail++ } else println("ok   $name")
    }

    // ---- round-trip happy path ----
    run {
        val data = ByteArray(1024) { (it * 31).toByte() }
        val wire = FileChunk.encode("abcd1234", 7, data)
        val dec  = FileChunk.decode(wire) ?: error("decode null")
        check("happy id",   dec.id == "abcd1234")
        check("happy seq",  dec.seq == 7)
        check("happy data", dec.data.contentEquals(data))
    }

    // ---- big seq number ----
    run {
        val wire = FileChunk.encode("x", 0x7fffffff, byteArrayOf())
        val dec  = FileChunk.decode(wire)!!
        check("max seq", dec.seq == 0x7fffffff && dec.id == "x" && dec.data.isEmpty())
    }

    // ---- empty data ----
    run {
        val wire = FileChunk.encode("id", 0, ByteArray(0))
        val dec  = FileChunk.decode(wire)!!
        check("empty data", dec.data.isEmpty())
    }

    // ---- malformed: too short ----
    check("too short", FileChunk.decode(byteArrayOf()) == null)
    check("short 4 bytes", FileChunk.decode(byteArrayOf(1, 'a'.code.toByte(), 0, 0)) == null)

    // ---- malformed: zero id_len ----
    check("zero id_len", FileChunk.decode(byteArrayOf(0, 0, 0, 0, 0)) == null)

    // ---- malformed: id_len overruns ----
    check("id_len overruns", FileChunk.decode(byteArrayOf(50, 'a'.code.toByte())) == null)

    // ---- max chunk size produces sensible wire size ----
    run {
        val big = ByteArray(FileChunk.MAX_CHUNK_BYTES) { 0x42 }
        val wire = FileChunk.encode("abcdefgh", 1, big)
        val expected = 1 + 8 + 4 + FileChunk.MAX_CHUNK_BYTES
        check("max-chunk wire size", wire.size == expected)
        val dec = FileChunk.decode(wire)!!
        check("max-chunk round-trip", dec.data.contentEquals(big))
    }

    // ---- id over MAX_ID_BYTES rejected at encode ----
    run {
        var threw = false
        try { FileChunk.encode("x".repeat(FileChunk.MAX_ID_BYTES + 1), 0, ByteArray(0)) }
        catch (_: IllegalArgumentException) { threw = true }
        check("over-long id rejected", threw)
    }

    if (fail > 0) { println("\n$fail failures"); kotlin.system.exitProcess(1) }
    println("\nALL OK")
}
