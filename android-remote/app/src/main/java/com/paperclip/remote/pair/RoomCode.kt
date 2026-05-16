package com.paperclip.remote.pair

import java.security.SecureRandom

/**
 * 6-character room code. Base32 with the ambiguous characters removed
 * (0/1/I/O/L/U). Matches the relay's regex check.
 */
object RoomCode {
    private const val ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"
    private val rng = SecureRandom()

    fun generate(): String = buildString(6) {
        repeat(6) { append(ALPHABET[rng.nextInt(ALPHABET.length)]) }
    }

    fun normalize(raw: String): String? {
        val cleaned = raw.uppercase().filter { it in ALPHABET }
        return if (cleaned.length == 6) cleaned else null
    }
}
