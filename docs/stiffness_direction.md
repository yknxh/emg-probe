# Stiffness-direction experiment (forearm + upper-arm band)

This experiment asks how the EMG pattern changes in a tool task such as wiping, depending on:

- **which direction the subject stiffens the arm:** pressing direction z vs surface directions xy;
- **surface orientation:** horizontal vs vertical;
- **contact:** a real wipe vs the same motion in the air.

One donning runs **① a quick muscle map (~5 min) → ② 10 trials per scenario → ③ an automatic summary**.

> Only EMG is measured. "Stiffness" is an instruction to the subject. Measuring actual stiffness needs mechanical
> perturbations.

Band placement: see [muscle map § 1](muscle_map.md#1-where-to-wear-the-bands).

## Scenarios

Keep these the same in every scenario:

- the same tool and the same grip;
- the same standing position (mark the feet).

Mark a **3 cm × 30 cm lane** with tape. Wipe **left–right** along it, one stroke per pace beep (default 1 s). If
possible, put a scale under the horizontal board and note the pressing force after the trial (`n 3kg`).

The design is a 2 × 2 per surface — **z stiffness (high/low) × xy stiffness (high/low)** — plus an **air wipe**
(no contact) control.

| id | Setup | Instruction (summary) | What it shows / expectation |
|---|---|---|---|
| `H_air` | horizontal | Same motion 2–3 cm above the lane, no contact | EMG of the motion alone (no contact, no force) |
| `H_light` | horizontal | Touch lightly, relaxed | **Reference wipe** (low z and xy stiffness) |
| `H_stiff_z` | horizontal | Stiff in the vertical direction: press hard and steady; path accuracy doesn't matter | Pressing → triceps ↑, wrist co-contraction ↑ |
| `H_stiff_xy` | horizontal | Follow the lane exactly; press lightly | Path precision → wrist radial/ulnar and flexor/extensor co-contraction ↑; triceps below `H_stiff_z` |
| `H_stiff_all` | horizontal | Whole arm stiff: press hard and follow the lane | Upper bound (full co-contraction) |
| `V_light` | vertical | Touch the wall lightly, relaxed | Gravity direction changes → activity to hold the arm up (radial wrist, biceps ↑) |
| `V_stiff_z` | vertical | Stiff toward the wall: push hard; path doesn't matter | Pushing = elbow extension → triceps ↑ |
| `V_stiff_xy` | vertical | Follow the lane exactly; push lightly | xy stiffness in a vertical plane |
| `V_air` *(optional)* | vertical | Motion only, 2–3 cm in front of the wall | Vertical air control |
| `V_stiff_all` *(optional)* | vertical | Push hard and follow the lane | Vertical upper bound |
| `H_press` *(optional)* | horizontal | Press with the `H_stiff_z` force, don't move | Separates pressing force from motion |
| `H_light_y` *(optional)* | horizontal | `H_light`, but wiping forward–back | Stroke direction (mostly elbow flexion/extension) |

- **Core set:** the first 8 rows, 10 trials each, 80 trials in total.
  - `--scenarios all` runs all 12.
  - `--scenarios H_light,H_stiff_z,...` runs a subset.
  - `--list` prints the full on-screen cues (in Korean).
- **Contrasts** computed in the summary:
  - z vs xy stiffness (horizontal and vertical);
  - real vs air wipe;
  - vertical vs horizontal;
  - full stiffness vs light;
  - press-and-wipe vs press only;
  - forward–back vs left–right.
- **Confound:** `H_stiff_z` also raises the normal force, so "z stiffness" and "pressing force" are mixed. Note the
  scale reading, and use `H_press` to see the effect of force alone.
- **Order:** the default, `interleaved`, runs the horizontal setup first and then the vertical one. Within a setup,
  each round runs every scenario once in random order, so fatigue and electrode drift do not pile up on one scenario.
  Use another `--seed` for each subject. `--order blocked` runs one scenario at a time.
- **Duration:** mapping about 5 min, plus about 15 s per trial × 80 ≈ 30–40 min. Take a break every ~20 trials;
  nothing is recorded while the prompt waits.

## Running

```bash
emg-probe ports                                              # which port is which band
emg-probe direction --simulate --band wrist --band upperarm --trials 2 --scenarios H_light,H_stiff_z   # practice
emg-probe direction --band wrist=/dev/cu.usbmodemAAA --band upperarm=/dev/cu.usbmodemBBB \
    --note "dk; forearm crease-5cm box ulnar, upper arm acromion-58% box lateral; circumference 26/29 cm"
```

1. **Mapping.** Follow the prompts; a channel → muscle table is printed at the end. Then:
   - ENTER: use this map
   - `m`: map again
   - `q`: quit
2. **Trials.** The next scenario and its cue are shown; press ENTER.
   - The trial runs: 2 s *pre* (hold the tool still at the start), start sound, 8 s *task* (one stroke per beep),
     stop sound, 1 s *post* (relax).
   - The live line shows each band's mean `norm`.
   - A `##### 셋업` banner announces the switch from horizontal to vertical.
3. **Between trials:**
   - ENTER: next trial
   - `r`: discard the last trial and redo it (its folder stays, marked `discarded`)
   - `n TEXT`: add a note to the last trial (e.g. `n 4kg left lane once`)
   - `q`: save and quit

   Ctrl+C during a trial aborts that trial only, and the same trial comes up again.
4. **Resume** after a break or an interruption:
   ```bash
   emg-probe direction --resume recordings/stiffness_direction_YYYYmmdd_HHMMSS --band wrist=... --band upperarm=...
   ```
   The map is reused, which is fine if the bands stayed on. If they came off, start a new experiment so it maps again.

| Option | Meaning |
|---|---|
| `--list` | Print scenarios and cues |
| `--trials 10` | Trials per scenario |
| `--task-s 8 --pre-s 2 --post-s 1` | Phase lengths (s) |
| `--pace-s 1` | Pace beep period; 0 = off |
| `--quiet` | No sounds |
| `--mapping DIR` | Reuse a map made in this donning (a simulated map is refused for real bands) |
| `--map-reps 3` | Mapping repetitions |
| one `--band` only | One band |

## Output

```
recordings/stiffness_direction_YYYYmmdd_HHMMSS/
  experiment.json            settings, scenario definitions, plan, trial entries (status, note, discarded)
  mapping/                   muscle_map.json / .md / .png + wrist/, upperarm/ raw data
  trials/001_H_light_t01/    one trial = one session
      metadata.json, analysis.json
      wrist/ upperarm/       emg.csv, raw.bin, imu.csv, events.jsonl, features.jsonl (100 ms RMS)
  trials.csv                 one row per valid trial × band: level, g_<group>, norm1-8, rms1-8, IMU, note
  summary.md / summary.json  per-scenario medians + contrasts (AUC)
  summary.png                scenario × channel heatmap (W1-8, U1-8)
```

- **Trial value:** the median of the 100 ms RMS windows in the *task* phase, without its first 1 s and last 0.5 s.
  *pre* and *post* are recorded too, if another window is wanted.
- **`level`:** mean `norm` over the band's responsive channels. **`g_<group>`:** mean `norm` of the channels mapped to
  one muscle group (e.g. `g_triceps`, `g_flexors`).
- **AUC:** P(a trial of A > a trial of B). 0.5 means no difference; ≥ 0.8 or ≤ 0.2 means a difference that is
  consistent across the 10 trials.
- **IMU:** `acc_x/y/z` (mean gravity direction) confirms the forearm posture of each setup. `gyro_rms_rad_s` checks
  that the amount of motion was similar across scenarios. If motion differs, it can explain an EMG difference.
- **`hum_max`:** the largest per-channel 60 Hz harmonic fraction. Above 0.3, check contact and cables.
- **Recompute:**
  ```bash
  emg-probe direction --analyse recordings/stiffness_direction_...
  emg-probe map --analyse recordings/stiffness_direction_.../mapping   # then re-run direction --analyse
  ```

## Interpretation

- **Channel → muscle is approximate** because of crosstalk, especially on the forearm.
- **`norm` is comparable within one donning only.** Across days or subjects, compare ratios between scenarios or AUCs.
- **EMG is not stiffness.** More co-contraction usually means stiffer joints, but EMG is not a stiffness value.
  The z condition is also confounded with normal force. Direction-specific stiffness needs perturbations.
- **Trials have a fixed length** with sound cues and no exact contact marks. For contact transitions, use the
  [contact log](contact_log.md).
