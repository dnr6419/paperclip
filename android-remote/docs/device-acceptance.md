# Device acceptance — SEED §11 DoD

The five must-pass metrics in [`SEED.md §11`](../SEED.md) can only be
measured on real hardware. This file is the **template + ledger**:
each test run records its setup, the numbers, and a pass/fail call
under a dated heading so future regressions have a baseline to compare
against.

The branch ships as **mechanically green** (46 automated cases pass)
but is **not** considered DoD-green until at least one row in the
ledger below has all five metrics in the green column.

## How to run a session

1. Pick a hardware pair (one controller phone, one controlled phone).
2. Pick a relay deployment (`server/` running somewhere reachable).
3. On the controlled phone: enable Accessibility for paperclip-remote
   (Settings → Accessibility → paperclip-remote → On).
4. Pair the two phones via QR + safety-code (first time) or via the
   peer list (subsequent).
5. Run each measurement below. Record the numbers in a new row of the
   ledger.

## Metrics

| Metric | Target | Stretch | How to measure |
|---|---|---|---|
| **M1. Input → on-screen feedback** | ≤ 500 ms median | ≤ 300 ms | Open a stopwatch app (e.g. Google's *Clock → Stopwatch*) on the **controlled** phone. Position the **controller**'s VideoSurface next to it. Film both phones at 60 fps with a third camera. Tap on the controller and start the stopwatch in the same motion (a single fingertip crossing both phones works for visual sync). Count frames from "tap completes" to "stopwatch updates" on the controlled phone, divided by 60. Repeat ≥ 10 trials, take the median. |
| **M2. Video frame rate** | ≥ 20 fps | ≥ 30 fps | Play [Big Buck Bunny 30 fps test pattern](https://test-videos.co.uk/) on the controlled phone. Observe the controller's VideoSurface — are frames visibly choppy? More objectively, `adb logcat -s H264Encoder` while sharing shows one log line per emitted access unit; over a 10-second window count entries / 10. |
| **M3. Cold-start to first frame** | ≤ 8 s | ≤ 5 s | From the controlled phone's *Ready* screen, start a stopwatch and tap **Start sharing**. Stop the stopwatch when the controller's VideoSurface shows the first non-black frame. |
| **M4. Pairing time (code → connected)** | ≤ 15 s | ≤ 10 s | Reset both apps' state (clear app data). On the controlled phone, start the stopwatch and tap **Start pairing**. Stop when both phones reach *Ready* (i.e. after Accept on both, which is when the safety-code comparison ends). |
| **M5. 1 GiB file transfer** | < 5 min | < 3 min | Generate `dd if=/dev/urandom of=/tmp/test1gib bs=1M count=1024` on a PC, transfer via USB to the controller phone, push from the *Send file* button. Time from selection to "saved to …" on the controlled phone. |

## Recording a run

Append a new section below. Keep the existing rows — comparing
regressions across hardware is the whole point of the ledger.

### Template (copy-paste)

```
### YYYY-MM-DD — <pair description> — <pass/fail>

- **Controller**: <brand model, Android version, Wi-Fi band>
- **Controlled**: <brand model, Android version, Wi-Fi band>
- **Relay**: <host, region, TLS yes/no>
- **Network**: <RTT controller→relay, controlled→relay, both in ms>
- **App build**: `<commit sha or version>`

| Metric | Target | Stretch | Measured | Pass? |
|---|---|---|---|---|
| M1 input → feedback (median, n=10) | ≤ 500 ms | ≤ 300 ms | __ ms | ☐ |
| M2 frame rate (10 s window) | ≥ 20 fps | ≥ 30 fps | __ fps | ☐ |
| M3 cold start → first frame | ≤ 8 s | ≤ 5 s | __ s | ☐ |
| M4 pairing time | ≤ 15 s | ≤ 10 s | __ s | ☐ |
| M5 1 GiB file transfer | < 5 min | < 3 min | __ min __ s | ☐ |

Notes: <anything unusual — auto-resume kicked in twice, bitrate slider
exercised, etc.>
```

## Ledger

_None yet. First on-device run will land here as part of DOF-R-15._
