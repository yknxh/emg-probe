# emg-probe

Recording and test protocols for the Waveletech 8-channel EMG band (wired, 2 kHz, 8 EMG channels + IMU).
Use one band, or two bands at once (forearm + upper arm). Every protocol runs without hardware using `--simulate`.

What it covers:

- Live view and a 30 s signal check of one band
- A quick **channel → muscle map** for one or two bands
- **Contact / phase logging** with ENTER marks
- The **stiffness-direction** scenario experiment (mapping, then N trials per scenario)
- **Gesture** classification and subjective **effort** levels (GUI)

EMG is recorded in filtered counts. Nothing here measures force or stiffness; labels such as "stiff" or "medium effort" are
instructions to the subject.

## Install

Requires Python 3.9+. Developed on macOS. The GUI protocols (`view`, `gesture`, `effort`) also need a display.

```bash
git clone https://github.com/yknxh/emg-probe.git && cd emg-probe
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
emg-probe            # list commands
```

On-screen cues for the subject are in Korean. If they render as boxes in the GUIs, install a Korean font such as
AppleGothic or Noto Sans CJK KR.

## Commands

| Command | What it does | Guide |
|---|---|---|
| `emg-probe ports` | List USB serial ports. Unplug and re-plug one band to see which port is which. | |
| `emg-probe view --port P` | Live plot of raw EMG (8 ch) and IMU (6 axes) | [hardware](docs/hardware.md) |
| `emg-probe check --port P` | 10 s rest / 10 s fist / 10 s rest. Prints per-channel fist/rest ratio and 60 Hz share. | [hardware](docs/hardware.md) |
| `emg-probe map --band wrist=P1 --band upperarm=P2` | Channel → muscle map, about 5 min. Also catches swapped ports. | [muscle map](docs/muscle_map.md) |
| `emg-probe log --band wrist=P1 [--mapping DIR]` | Free-form logging. ENTER toggles free/contact, or steps through `--phases`. | [contact log](docs/contact_log.md) |
| `emg-probe direction --band wrist=P1 --band upperarm=P2` | Stiffness-direction experiment: mapping, scenarios × 10 trials, summary | [stiffness direction](docs/stiffness_direction.md) |
| `emg-probe gesture --port P` | Rest / fist / open: guided collection, held-out test, live prediction | [gesture & effort](docs/gesture_effort.md) |
| `emg-probe effort --port P` | Rest / low / medium / higher effort, plus slow ramps | [gesture & effort](docs/gesture_effort.md) |
| `emg-probe timing SESSION` | Per-band clock fit of a saved two-band session | [hardware](docs/hardware.md) |

`emg-probe COMMAND -h` shows every option. `python -m emgprobe COMMAND` works without installing the script.

### Bands and ports

- `--band ROLE=PORT` names a band. `ROLE` is `wrist` (forearm band) or `upperarm`. Give `--band` once for one band or twice for two.
  With `--simulate` the port can be left out (`--band wrist --band upperarm`).
- The single-band tools (`view`, `check`, `gesture`, `effort`) take `--port`. Without it they pick the only USB serial
  port, if there is exactly one.
- Close other programs that hold the port before recording.

## Typical session (two bands)

```bash
emg-probe direction --simulate --band wrist --band upperarm --trials 2 --scenarios H_light,H_stiff_z   # practice, no hardware
emg-probe ports
emg-probe check --port /dev/cu.usbmodemAAA                    # each band: contact OK?
emg-probe direction --band wrist=/dev/cu.usbmodemAAA --band upperarm=/dev/cu.usbmodemBBB \
    --note "forearm crease-5cm box ulnar; upper arm 58% box lateral"
```

`direction` runs the muscle map first. To log free-form in the same donning, reuse that map:

```bash
emg-probe log --band wrist=... --band upperarm=... --mapping recordings/stiffness_direction_.../mapping
```

A muscle map is valid only for the donning it was recorded in. If a band is taken off or shifts, map again.

## Recordings

Sessions go to `./recordings/<kind>_YYYYmmdd_HHMMSS/`. Use `--output` to change the location. Every band folder has
the same files:

| File | Content |
|---|---|
| `raw.bin` | Every byte read from the port |
| `emg.csv` | Per 2 kHz sample: PC time, phase event, counter, stream segment, raw and filtered counts for ch1–8 |
| `imu.csv` | Gyro (rad/s), accelerometer (m/s²), gyro minus calibrated bias |
| `events.jsonl` | Phase marks: `event_id`, `host_time_s`, `phase`, `round`, `label` |
| `features.jsonl` | RMS windows of the filtered EMG, each with its phase, time and 60 Hz harmonic fraction |
| `metadata.json` | Settings, filter, status, counts, errors, note |

Two-band sessions have `<session>/wrist/` and `<session>/upperarm/`, plus a session-level `metadata.json`.
Protocol outputs such as `analysis.json`, `muscle_map.*`, `summary.*` and `windows.csv` are described in each guide.
Interrupted sessions keep their raw data.

## Signal processing

- **Filter.** Causal, single pass. 20–450 Hz Butterworth band-pass (order 4), plus notches at every 60 Hz harmonic
  from 60 to 420 Hz (4 Hz wide, each applied three times). The filter is reset on a counter gap, a stream segment
  change, or a PC read gap over 0.25 s. The first 1 s after a reset is not used. Settings are stored in every
  `metadata.json`, and maps are rejected if they differ.
- **Features.** RMS of the filtered signal: 100 ms windows with a 100 ms stride for the terminal protocols, 250 ms
  windows with a 100 ms stride for gesture/effort. Units are counts, not µV.
- **Timing.** PC read-completion times on one monotonic clock shared by the bands, with about 10–30 ms jitter.
  This is fine for 100 ms RMS, but not for millisecond onset differences between bands. See [hardware](docs/hardware.md).

## Repository layout

```
src/emgprobe/
  device.py        frame parser, serial reader, simulated frames, port lookup
  filters.py       causal EMG filter and spectral metrics
  recorder.py      Recorder thread (files above); WindowRecorder = 100 ms windows
  bands.py         several bands side by side, clock fit, window merge
  analysis.py      phase windows, muscle-map normalization, IMU motion, AUC
  terminal.py      timed phases with a live line, sounds, prompts
  files.py         JSON helpers, session folders
  cli.py           `emg-probe` command
  protocols/       view, check, muscle_map, contact_log, direction, gesture, effort
tests/             simulated end-to-end tests
docs/              guides
```

To add a protocol: subclass `WindowRecorder` (set `FEATURE_PHASES` and override `sim_gain` for the simulation). Drive
phases with `Bands.set_phase` or `terminal.timed`. Then add a `main(argv)` and register it in `cli.COMMANDS`.

## Tests

```bash
MPLBACKEND=Agg python -m unittest
```

The tests run every protocol on simulated bands, about 2–3 minutes in all. They check file layout, protocol flow and
analysis. They do not verify sensor accuracy or electrical safety.
