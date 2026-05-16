/*
 * Cross-check that the JVM-side crypto math matches docs/protocol_reference.py.
 * Reads fixed test vectors from stdin (one base16 value per line) and
 * prints back the derived values for the Python harness to assert.
 *
 * Inputs (in order):
 *   1. a_id_priv      (32 hex bytes)
 *   2. b_id_priv      (32 hex bytes)
 *   3. a_eph_priv     (32 hex bytes)
 *   4. b_eph_priv     (32 hex bytes)
 *   5. canonical_hello_a  (hex of UTF-8 bytes)
 *   6. canonical_hello_b  (hex of UTF-8 bytes)
 *   7. plaintext      (hex)
 *
 * Outputs (one per line, all hex / strings):
 *   a_id_pub
 *   b_id_pub
 *   safety_code
 *   transcript
 *   a_send_key
 *   a_recv_key
 *   b_send_key
 *   b_recv_key
 *   wire_a_text_counter0          (TEXT frame, counter=0, key=a_send)
 *   wire_b_binary_counter42_tag01  (BINARY frame, counter=42, tag=0x01, key=b_send)
 *   decrypted_a_text              (b_recv unwraps wire_a_text)
 *   decrypted_b_binary            (a_recv unwraps wire_b_binary)
 */
import java.nio.charset.StandardCharsets;
import java.security.KeyFactory;
import java.security.MessageDigest;
import java.security.interfaces.XECPrivateKey;
import java.security.interfaces.XECPublicKey;
import java.security.spec.NamedParameterSpec;
import java.security.spec.XECPrivateKeySpec;
import java.security.spec.XECPublicKeySpec;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Scanner;
import javax.crypto.Cipher;
import javax.crypto.KeyAgreement;
import javax.crypto.Mac;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

public class CryptoCrossCheck {

    // ---- hex helpers --------------------------------------------------------
    static String hex(byte[] b) {
        StringBuilder sb = new StringBuilder(b.length * 2);
        for (byte x : b) sb.append(String.format("%02x", x));
        return sb.toString();
    }
    static byte[] unhex(String s) {
        byte[] out = new byte[s.length() / 2];
        for (int i = 0; i < out.length; i++)
            out[i] = (byte) Integer.parseInt(s.substring(i*2, i*2+2), 16);
        return out;
    }

    // ---- X25519 -------------------------------------------------------------
    static byte[] x25519Pub(byte[] priv) throws Exception {
        // RFC 7748 scalar -> public via base point. JDK exposes this via
        // a sentinel: build a private key and derive its public.
        NamedParameterSpec params = NamedParameterSpec.X25519;
        KeyFactory kf = KeyFactory.getInstance("XDH");
        XECPrivateKey privKey = (XECPrivateKey) kf.generatePrivate(new XECPrivateKeySpec(params, priv));
        // JDK's XECPrivateKey can compute the public scalar by doing scalarMult(priv, basepoint).
        // The cleanest API: use KeyAgreement to compute pub = priv * basepoint by
        // agreeing with the "all-zero except 9" basepoint.
        // Easier: just use KeyPairGenerator with a fixed scalar via XECPrivateKeySpec,
        // then call ((XECPrivateKey) k).getScalar() — but public derivation isn't on
        // XECPrivateKey directly. Workaround: KeyAgreement with the standard basepoint.
        byte[] basepoint = new byte[32]; basepoint[0] = 9;
        XECPublicKey bp = (XECPublicKey) kf.generatePublic(new XECPublicKeySpec(params, decodeLE(basepoint)));
        KeyAgreement ka = KeyAgreement.getInstance("XDH");
        ka.init(privKey);
        ka.doPhase(bp, true);
        return ka.generateSecret();
    }

    static java.math.BigInteger decodeLE(byte[] le) {
        byte[] be = new byte[le.length];
        for (int i = 0; i < le.length; i++) be[i] = le[le.length - 1 - i];
        return new java.math.BigInteger(1, be);
    }

    static byte[] x25519Dh(byte[] priv, byte[] peerPub) throws Exception {
        NamedParameterSpec params = NamedParameterSpec.X25519;
        KeyFactory kf = KeyFactory.getInstance("XDH");
        XECPrivateKey privKey = (XECPrivateKey) kf.generatePrivate(new XECPrivateKeySpec(params, priv));
        XECPublicKey pubKey = (XECPublicKey) kf.generatePublic(new XECPublicKeySpec(params, decodeLE(peerPub)));
        KeyAgreement ka = KeyAgreement.getInstance("XDH");
        ka.init(privKey);
        ka.doPhase(pubKey, true);
        return ka.generateSecret();
    }

    // ---- HKDF-SHA256 -------------------------------------------------------
    static byte[] hmacSha256(byte[] key, byte[] data) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(key, "HmacSHA256"));
        return mac.doFinal(data);
    }
    static byte[] hkdf(byte[] salt, byte[] ikm, byte[] info, int length) throws Exception {
        byte[] prk = hmacSha256(salt, ikm);
        // For length <= 32 we need only T(1).
        byte[] t1Input = new byte[info.length + 1];
        System.arraycopy(info, 0, t1Input, 0, info.length);
        t1Input[info.length] = 0x01;
        byte[] t1 = hmacSha256(prk, t1Input);
        return Arrays.copyOf(t1, length);
    }

    // ---- safety code -------------------------------------------------------
    static String safetyCode(byte[] a, byte[] b) throws Exception {
        byte[] lo, hi;
        if (compareLex(a, b) <= 0) { lo = a; hi = b; } else { lo = b; hi = a; }
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        md.update("paperclip-remote v2 safety".getBytes(StandardCharsets.UTF_8));
        md.update(lo); md.update(hi);
        String h = hex(md.digest()).substring(0, 16).toUpperCase();
        return h.substring(0,4) + "-" + h.substring(4,8) + "-" + h.substring(8,12) + "-" + h.substring(12,16);
    }
    static int compareLex(byte[] a, byte[] b) {
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            int ai = a[i] & 0xff, bi = b[i] & 0xff;
            if (ai != bi) return ai - bi;
        }
        return a.length - b.length;
    }

    // ---- transcript hash ---------------------------------------------------
    static byte[] transcriptHash(byte[] a, byte[] b) throws Exception {
        byte[] lo, hi;
        if (lengthThenLexLe(a, b)) { lo = a; hi = b; } else { lo = b; hi = a; }
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        md.update(lo); md.update(hi);
        return md.digest();
    }
    static boolean lengthThenLexLe(byte[] a, byte[] b) {
        if (a.length != b.length) return a.length < b.length;
        return compareLex(a, b) <= 0;
    }

    // ---- session-key derivation -------------------------------------------
    static byte[] hkdfRoleKey(byte[] transcript, byte[] ikm, String role) throws Exception {
        return hkdf(transcript, ikm, role.getBytes(StandardCharsets.UTF_8), 32);
    }

    // ---- AEAD wrap ---------------------------------------------------------
    static byte[] nonceFor(long counter) {
        byte[] n = new byte[12];
        for (int i = 0; i < 8; i++) n[4 + i] = (byte) ((counter >>> ((7 - i) * 8)) & 0xff);
        return n;
    }

    static byte[] wrap(byte[] key, long counter, int kind, Integer tag, byte[] pt) throws Exception {
        int hdrLen = (kind == 0x42) ? 10 : 9;
        byte[] hdr = new byte[hdrLen];
        for (int i = 0; i < 8; i++) hdr[i] = (byte) ((counter >>> ((7 - i) * 8)) & 0xff);
        hdr[8] = (byte) kind;
        if (kind == 0x42) hdr[9] = tag.byteValue();
        Cipher c = Cipher.getInstance("ChaCha20-Poly1305");
        c.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, "ChaCha20"), new IvParameterSpec(nonceFor(counter)));
        c.updateAAD(hdr);
        byte[] ct = c.doFinal(pt);
        byte[] wire = new byte[hdr.length + ct.length];
        System.arraycopy(hdr, 0, wire, 0, hdr.length);
        System.arraycopy(ct, 0, wire, hdr.length, ct.length);
        return wire;
    }

    static byte[] unwrap(byte[] key, byte[] wire) throws Exception {
        long counter = 0;
        for (int i = 0; i < 8; i++) counter = (counter << 8) | (wire[i] & 0xffL);
        int kind = wire[8] & 0xff;
        int hdrLen = (kind == 0x42) ? 10 : 9;
        byte[] hdr = Arrays.copyOfRange(wire, 0, hdrLen);
        byte[] ct = Arrays.copyOfRange(wire, hdrLen, wire.length);
        Cipher c = Cipher.getInstance("ChaCha20-Poly1305");
        c.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, "ChaCha20"), new IvParameterSpec(nonceFor(counter)));
        c.updateAAD(hdr);
        return c.doFinal(ct);
    }

    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(System.in);
        byte[] aIdPriv  = unhex(sc.nextLine().trim());
        byte[] bIdPriv  = unhex(sc.nextLine().trim());
        byte[] aEphPriv = unhex(sc.nextLine().trim());
        byte[] bEphPriv = unhex(sc.nextLine().trim());
        byte[] helloA   = unhex(sc.nextLine().trim());
        byte[] helloB   = unhex(sc.nextLine().trim());
        byte[] pt       = unhex(sc.nextLine().trim());

        byte[] aIdPub  = x25519Pub(aIdPriv);
        byte[] bIdPub  = x25519Pub(bIdPriv);
        byte[] aEphPub = x25519Pub(aEphPriv);
        byte[] bEphPub = x25519Pub(bEphPriv);

        String code = safetyCode(aIdPub, bIdPub);
        byte[] transcript = transcriptHash(helloA, helloB);

        byte[] staticDh_a = x25519Dh(aIdPriv, bIdPub);
        byte[] ephDh_a    = x25519Dh(aEphPriv, bEphPub);
        byte[] ikmA = concat("paperclip-remote v2 session-key".getBytes(StandardCharsets.UTF_8),
                             transcript, staticDh_a, ephDh_a);
        byte[] aSend = hkdfRoleKey(transcript, ikmA, "controller");
        byte[] aRecv = hkdfRoleKey(transcript, ikmA, "controlled");

        byte[] staticDh_b = x25519Dh(bIdPriv, aIdPub);
        byte[] ephDh_b    = x25519Dh(bEphPriv, aEphPub);
        byte[] ikmB = concat("paperclip-remote v2 session-key".getBytes(StandardCharsets.UTF_8),
                             transcript, staticDh_b, ephDh_b);
        byte[] bSend = hkdfRoleKey(transcript, ikmB, "controlled");
        byte[] bRecv = hkdfRoleKey(transcript, ikmB, "controller");

        byte[] wireA = wrap(aSend, 0L, 0x54, null, pt);
        byte[] wireB = wrap(bSend, 42L, 0x42, 0x01, pt);
        byte[] decA = unwrap(bRecv, wireA);
        byte[] decB = unwrap(aRecv, wireB);

        System.out.println(hex(aIdPub));
        System.out.println(hex(bIdPub));
        System.out.println(code);
        System.out.println(hex(transcript));
        System.out.println(hex(aSend));
        System.out.println(hex(aRecv));
        System.out.println(hex(bSend));
        System.out.println(hex(bRecv));
        System.out.println(hex(wireA));
        System.out.println(hex(wireB));
        System.out.println(hex(decA));
        System.out.println(hex(decB));
    }

    static byte[] concat(byte[]... parts) {
        int n = 0; for (byte[] p : parts) n += p.length;
        byte[] out = new byte[n]; int o = 0;
        for (byte[] p : parts) { System.arraycopy(p, 0, out, o, p.length); o += p.length; }
        return out;
    }
}
