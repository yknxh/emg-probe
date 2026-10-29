# Contact / phase logging

`emg-probe log` records one or two bands freely while the operator marks phases with ENTER. Without a map it records
filtered EMG and per-window channel RMS. With a [muscle map](muscle_map.md) from the same donning, each window also gets
`norm`, `level` and per-group values.

```bash
emg-probe log --simulate --band wrist --band upperarm                      # practice
emg-probe log --band wrist=/dev/cu.A --band upperarm=/dev/cu.B \
    --mapping recordings/muscle_map_YYYYmmdd_HHMMSS --note "surface, posture"
```

## Controls

| Input | Effect |
|---|---|
| ENTER | Toggle free ↔ contact (contact = hand on the surface and pressing) |
| `q` + ENTER or Ctrl+C | Stop, save, analyse |

The live line shows each band's mapping `level`, or the median RMS without a map. `EMG STALE` appears when no data has
arrived for 1 s.

Contact marks are manual, so they lag the true contact by about 0.2–0.3 s. Force is not recorded.

### Fixed phase sequence (`--phases`)

```bash
emg-probe log --band wrist=... --band upperarm=... --phases approach,grasp,contact,free
```

The session starts in the first phase, and each ENTER moves to the next. ENTER in the last phase ends the session.
`label` is the phase index, and the `contact` column is 1 only in the `contact` phase.

### Many sessions in a row (`--sessions`)

```bash
emg-probe log --band wrist=... --band upperarm=... --sessions 20 --mapping ...
```

- Sessions are saved as `recordings/contact_batch_.../session_01 … session_20`, each analysed on its own.
- ENTER starts the next session. `q` + ENTER (or Ctrl+C) ends the batch.
- If a session fails (e.g. `EMG startup timeout`), the batch stops. Finished sessions are kept, so start a new batch
  for the rest.

## Output

```
contact_log_.../                (or contact_batch_.../session_NN/)
  metadata.json                 settings, phases, bands (port, USB location, per-band counts)
  muscle_map.json               copy of the map used (if any)
  analysis.json                 per band and phase: windows, median RMS, median level and groups; merge; clock fit
  bands_windows.csv             each window of the first band joined to the nearest window of the other band (<band>_dt_s = offset)
  wrist/  upperarm/             emg.csv, imu.csv, events.jsonl, features.jsonl, windows.csv, metadata.json, raw.bin
```

`windows.csv` has, per 100 ms window:

- `time_s`, `event_id`, `phase`, `contact`, `contact_index`, `segment`, `rms1-8`;
- with a map, also `level`, `g_<group>` and `norm1-8`.

Recompute, optionally with another map:

```bash
emg-probe log --analyse recordings/contact_log_... [--mapping recordings/muscle_map_...]
```

Notes:

- A simulated map cannot be used with real bands, and a map made with other filter settings is refused.
- Normalized values are fractions of the mapping response. They are not MVC and not stiffness.
- Alignment between bands uses the PC clock. See [hardware § timing](hardware.md#timing-with-two-bands).
