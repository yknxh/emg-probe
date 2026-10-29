# Gesture and effort (GUI protocols)

Both protocols open a matplotlib window with Start/Ready and Stop & Save buttons. They use 250 ms RMS windows with a
100 ms stride, skipping the first 0.75 s of each hold. The subject's cues are in Korean.

## Gesture: rest / fist / open

```bash
emg-probe gesture --simulate
emg-probe gesture --port /dev/cu.A [--rounds 8]
emg-probe gesture --port /dev/cu.A --model recordings/gesture_.../model.json   # reuse a frozen model after re-donning
```

1. **Start.** 3 s preparation, then a 20 s rest baseline. The baseline also gives a gyro bias, applied only if the arm
   was still: gyro std < 0.02 rad/s and accel-norm std < 0.1 m/s².
2. **Rounds.** Each of the `--rounds` rounds (default 8, minimum 4) runs rest, fist and open once, in random order:
   3 s settle, 3 s hold, 5 s rest. Press Ready before each round.
3. **Model.** Trained on all rounds except the last two: standardized log-RMS with a regularized shared-covariance LDA.
   It is evaluated on the last two rounds only, and is not retrained afterwards.
4. **Live.** After the evaluation the window predicts live until Stop & Save.

Output:

- `model.json`, reusable with `--model`; all rounds then count as test data;
- `analysis.json`: window and trial accuracy, confusion matrix, rest false-activation rate, 60 Hz fraction;
- `report.md`.

Overlapping windows are not independent samples. A high score within one donning does not guarantee performance after
re-donning. The IMU is recorded but not used as a classifier input.

## Effort: subjective levels and ramps

```bash
emg-probe effort --simulate
emg-probe effort --port /dev/cu.A [--rounds 5] [--note "object, fit, posture"]
emg-probe effort --band wrist=/dev/cu.A --band upperarm=/dev/cu.B   # two bands
emg-probe effort --analyse recordings/effort_...
```

The subject holds the same object in the same posture:

1. 20 s rest baseline;
2. `--rounds` rounds of rest / low / medium / slightly strong effort, 4 s each, in random order;
3. one slow ramp up and one slow ramp down, 8 s each.

No maximum effort is asked for. Each step waits for Ready.

Output:

- `analysis.json`, `trial_summary.csv`;
- `report.md`: median RMS per level, between-trial CV, and which channels increase rest → low → medium → higher in
  each round;
- `analysis.png`.

Labels are instructed effort, not N or %MVC. `measured_force_N` and `measured_stiffness` are always `null`.
