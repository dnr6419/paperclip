package com.paperclip.remote.files

import android.content.Context
import android.net.Uri
import android.util.Log
import com.paperclip.remote.session.SessionHolder
import com.paperclip.remote.transport.BinaryTag
import com.paperclip.remote.transport.ControlMessage
import com.paperclip.remote.transport.SecureChannel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.security.SecureRandom

/**
 * Both ends of the FILE_BEGIN / FILE_DATA / FILE_END protocol. One
 * instance per session; subscribe to [transfers] for UI updates.
 *
 * Sender path: [sendFile] reads bytes from an InputStream in
 * MAX_CHUNK_BYTES pieces, emits one BINARY (tag 0x02) frame per piece,
 * brackets with FileBegin / FileEnd TEXT control frames.
 *
 * Receiver path: [bind] subscribes to the channel's incoming flow.
 * file_begin → open a file under context.externalCacheDir/received/
 * (no scoped-storage permission needed). file_data chunks write at
 * `seq * MAX_CHUNK_BYTES` so out-of-order arrivals correctly land in
 * place. file_end finalizes (or removes on ok=false).
 */
class FileTransferManager(
    private val context: Context,
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
) {
    sealed interface State {
        val id: String
        val name: String
        val total: Long

        data class Sending(override val id: String, override val name: String,
                           override val total: Long, val sent: Long) : State
        data class Receiving(override val id: String, override val name: String,
                             override val total: Long, val received: Long) : State
        data class Done(override val id: String, override val name: String,
                        override val total: Long, val path: String?) : State
        data class Failed(override val id: String, override val name: String,
                          override val total: Long, val reason: String) : State

        val progress: Float
            get() = when (this) {
                is Sending   -> if (total > 0) sent.toFloat() / total else 0f
                is Receiving -> if (total > 0) received.toFloat() / total else 0f
                is Done      -> 1f
                is Failed    -> 0f
            }
    }

    private val _transfers = MutableStateFlow<Map<String, State>>(emptyMap())
    val transfers: StateFlow<Map<String, State>> = _transfers.asStateFlow()

    private val incoming = mutableMapOf<String, IncomingState>()
    private val rng = SecureRandom()
    private var subscription: Job? = null

    private data class IncomingState(
        val name: String,
        val total: Long,
        val file: File,
        val out: FileOutputStream,
        var received: Long,
    )

    fun bind(channel: SecureChannel) {
        subscription?.cancel()
        subscription = scope.launch {
            channel.incoming.collect { plain ->
                when (plain) {
                    is SecureChannel.Plain.Text   -> handleText(plain.message)
                    is SecureChannel.Plain.Binary -> if (plain.tag == BinaryTag.FILE_DATA) handleData(plain.payload)
                }
            }
        }
    }

    fun unbind() {
        subscription?.cancel()
        subscription = null
        for ((_, st) in incoming) try { st.out.close() } catch (_: Exception) {}
        incoming.clear()
    }

    private fun handleText(msg: ControlMessage) {
        when (msg) {
            is ControlMessage.FileBegin -> beginReceive(msg)
            is ControlMessage.FileEnd   -> finishReceive(msg)
            else -> {}
        }
    }

    private fun beginReceive(b: ControlMessage.FileBegin) {
        if (incoming.containsKey(b.id)) return  // duplicate
        val safeName = b.name.replace(Regex("[^A-Za-z0-9._-]"), "_").take(120)
        val dir = File(context.externalCacheDir ?: context.cacheDir, "received").apply { mkdirs() }
        val target = File(dir, "${b.id}-$safeName")
        val out = try {
            FileOutputStream(target)
        } catch (e: Exception) {
            Log.e(TAG, "open output for ${b.id} failed", e)
            updateState(State.Failed(b.id, b.name, b.size, "open: ${e.message}"))
            return
        }
        incoming[b.id] = IncomingState(b.name, b.size, target, out, received = 0)
        updateState(State.Receiving(b.id, b.name, b.size, received = 0))
    }

    private fun handleData(payload: ByteArray) {
        val chunk = FileChunk.decode(payload) ?: return
        val st = incoming[chunk.id] ?: return
        try {
            val offset = chunk.seq.toLong() * FileChunk.MAX_CHUNK_BYTES
            st.out.channel.position(offset)
            st.out.write(chunk.data)
            st.received += chunk.data.size
            updateState(State.Receiving(chunk.id, st.name, st.total, st.received))
        } catch (e: Exception) {
            Log.e(TAG, "write chunk failed", e)
            try { st.out.close() } catch (_: Exception) {}
            st.file.delete()
            incoming.remove(chunk.id)
            updateState(State.Failed(chunk.id, st.name, st.total, "write: ${e.message}"))
        }
    }

    private fun finishReceive(e: ControlMessage.FileEnd) {
        val st = incoming.remove(e.id) ?: return
        try { st.out.close() } catch (_: Exception) {}
        if (!e.ok || (st.total > 0 && st.received < st.total)) {
            st.file.delete()
            updateState(State.Failed(e.id, st.name, st.total,
                if (!e.ok) "remote reported failure" else "incomplete"))
            return
        }
        updateState(State.Done(e.id, st.name, st.total, st.file.absolutePath))
    }

    /**
     * Stream a file out over the active session. The caller is
     * responsible for keeping `input` open until this coroutine
     * completes — typically open it from a content:// URI right before
     * the call, [stream] closes it in `finally`.
     */
    fun sendFile(stream: InputStream, name: String, size: Long, mime: String) {
        val channel = SessionHolder.get() ?: run {
            updateState(State.Failed(genId(), name, size, "no active session"))
            return
        }
        val id = genId()
        updateState(State.Sending(id, name, size, sent = 0))
        scope.launch {
            try {
                channel.sendText(ControlMessage.FileBegin(id, name, size, mime))
                val buf = ByteArray(FileChunk.MAX_CHUNK_BYTES)
                var seq = 0
                var sent = 0L
                stream.use { ins ->
                    while (true) {
                        val n = ins.read(buf)
                        if (n <= 0) break
                        val payload = FileChunk.encode(id, seq, buf.copyOfRange(0, n))
                        channel.sendBinary(BinaryTag.FILE_DATA, payload)
                        sent += n
                        seq += 1
                        updateState(State.Sending(id, name, size, sent))
                    }
                }
                channel.sendText(ControlMessage.FileEnd(id, ok = true))
                updateState(State.Done(id, name, size, path = null))
            } catch (e: Exception) {
                try { channel.sendText(ControlMessage.FileEnd(id, ok = false)) } catch (_: Exception) {}
                updateState(State.Failed(id, name, size, "send: ${e.message}"))
            }
        }
    }

    fun sendUri(uri: Uri, displayName: String, mime: String) {
        val resolver = context.contentResolver
        val size = resolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
        val stream = resolver.openInputStream(uri) ?: run {
            updateState(State.Failed(genId(), displayName, 0, "cannot open uri"))
            return
        }
        sendFile(stream, displayName, size, mime)
    }

    private fun updateState(s: State) {
        _transfers.value = _transfers.value + (s.id to s)
    }

    private fun genId(): String {
        val raw = ByteArray(8).also(rng::nextBytes)
        return raw.joinToString("") { "%02x".format(it) }
    }

    fun release() {
        unbind()
        scope.cancel()
    }

    companion object {
        private const val TAG = "FileTransferManager"
    }
}
