# Hardware, stream and timing

## The band

Waveletech 8-channel EMG wristband: a ring of 8 dry electrodes about 2.5 cm wide, plus a box holding the electronics
and IMU. It is connected over USB serial at 921600 baud and streams about 2 kHz per channel.

### Channel order

Channel order is not documented. Recordings made so far suggest that the channels run in order around the ring, with
the gap at the box between ch8 and ch1 (ch8–ch1 correlation is low). As a result:

- muscle directly under the box is poorly covered, so put the box over bone or over a border between muscles;
- check every donning with a [muscle map](muscle_map.md) rather than assuming the order.

## Frame format

Each frame is 29 bytes: `D2 D2 D2`, then a type byte, an 8-bit counter, and a 24-byte payload.

| Type | Payload |
|---|---|
| `AA` (EMG) | 8 × signed 24-bit big-endian counts, ch1–ch8 |
| `BB` (IMU) | 2 reserved bytes, then 3 × int16 gyro X/Y/Z (× 0.0012 → rad/s) and 3 × int16 accelerometer X/Y/Z (× 0.0005978 → m/s²), big-endian |

AA and BB frames share the counter. A step other than +1 is a dropped frame: the recorder counts it in
`counter_breaks`, starts a new stream `segment` and re-warms the filter. Bytes that do not belong to a frame are counted
in `discarded_bytes`. See `emgprobe/device.py`.

## Checking a band

```bash
emg-probe ports                       # which port is which: unplug/re-plug one band
emg-probe view --port /dev/cu.X       # raw counts; red dots = 24-bit limits (saturation)
emg-probe check --port /dev/cu.X      # 10 s rest, 10 s strong fist, 10 s rest
```

`check` prints, per channel:

- rest and fist RMS;
- the fist/rest ratio — `OK` if ≥ 2;
- the 60 Hz harmonic share of the raw signal at rest.

A channel that is `weak` on a fist, or has a high 60 Hz share, usually has poor skin contact. Tighten the band, wait
2–3 min, check again.

## Filtering and features

See `emgprobe/filters.py`. The filter is causal and runs in a single pass: a 20–450 Hz 4th-order Butterworth band-pass,
then notches at 60, 120, …, 420 Hz, each 4 Hz wide and applied three times.

Mains hum on this band can be over 1000× the resting EMG and drifts about 0.05 Hz off 60 Hz, hence the cascaded notches.

The first 1 s after a stream start or a reset is marked invalid and is not used. Settings (`filter.version` = 3) are
stored in every `metadata.json`. A muscle map made with other settings is refused.

Feature windows store the RMS of the filtered signal, plus `harmonic_fraction`: the share of 20–450 Hz power within
±4 Hz of a 60 Hz harmonic. It is a contact-quality indicator, not an SNR.

## Timing with two bands

- Each band has its own sample clock. The PC monotonic clock is the only shared time base: read-completion time, with
  about 10–30 ms jitter.
- Both recorders share one origin, and every phase mark is written to both bands with the same timestamp. So
  `event_id` and `host_time_s` line up across bands.
- `emg-probe timing SESSION` fits `host_time ≈ t0_s + sample / rate_hz` per band on its longest gap-free segment. It
  reports the clock error (ppm) and the PC read jitter. Use the fit to convert `sample` to time when you need
  sample-level alignment.
- This is enough for 100 ms RMS comparisons. Onset differences of a few ms between muscles on different bands cannot
  be trusted without hardware synchronization.
