# Band placement and the channel → muscle map

`emg-probe map` shows which muscle each channel roughly sits over, for one or two bands. The result is valid for one
donning only. `emg-probe direction` runs it as its first step.

## 1. Where to wear the bands

Rule: put the box (the gap between ch8 and ch1, see [hardware](hardware.md)) over bone or over a border between
muscles. That way the gap costs no muscle.

### Forearm band (`wrist`): upper third of the forearm, not the wrist

- **Height:** about 4–6 cm distal to the elbow crease (¼–⅓ of the forearm length), where the forearm is thickest.
- **Why:** Near the wrist there are only tendons and bone, so the signal is small and the electrodes slide whenever the
  wrist moves. The muscle bellies of the wrist flexors (FCR, PL, FCU), wrist extensors (ECRL, ECRB, EDC, ECU) and
  brachioradialis all sit in the upper third. One ring there covers the palm, dorsal, thumb and little-finger sides.
  Most 8-channel armband studies use this height.
- **Box:** on the **ulnar ridge**. This is the bone you can feel under the skin, running from the point of the elbow to
  the little-finger side of the wrist. The gap then lies on bone, with flexors on one side of the box and extensors on
  the other. If the box hits the table while wiping, rotate the band slightly toward the back of the hand.

```
Right forearm, cross-section (looking from the elbow toward the hand, palm down)
                     dorsal (up)
              ●  extensors EDC·ECRB  ●
   thumb    ●                          ●  ECU
   side     ●  brachioradialis          [box]  ← ulnar ridge (little-finger side)
   ECRL      ●                         ●  FCU
              ●  flexors FCR·PL·FDS ●
                     palm (down)
```

### Upper-arm band (`upperarm`): just below mid-arm, box on the outside

- **Height:** about **55–60 %** of the way from the acromion (shoulder tip) to the elbow crease, near the peak of the
  biceps. This spans two SENIAM sites: biceps at ⅓ of the acromion–cubital fossa line from the elbow, and triceps
  long/lateral head at 50 % of the acromion–olecranon line. Higher up you pick up deltoid; nearer the elbow there is
  only tendon.
- **Box:** **lateral midline**, on the border between biceps and triceps. The gap then falls on that border, and
  biceps (front) and triceps (back) each get 3–4 consecutive channels. The box also stays clear of the torso.
  With the box at the back, the gap lands in the middle of the triceps.
- The medial groove (torso side) carries nerves and vessels and has little muscle. One weak channel there is normal.

```
Right upper arm, cross-section (looking down from the shoulder)
                   front (biceps)
              ●  short head   long head  ●
  medial     ●                          ●  brachialis
 (torso)     groove                       [box]  ← lateral border
             ●                          ●
              ●  long head  lateral head ●
                   back (triceps)
```

### For every donning

- **One map per donning.** If a band comes off or shifts, map again.
- **Mark the skin** with a pen at the band edges and the box position. Record distances (cm), box direction and arm
  circumference in `--note`, so the placement can be repeated on another day or another subject.
- **Wait 2–3 min** after putting the bands on, so electrode contact settles. Make the band snug, but loose enough that
  the hand does not tingle. Tape the cables to the arm so they don't pull.
- **Shoulder muscles are not measured** (deltoid, pectoralis). Much of the in-plane force in tasks like wiping comes
  from the shoulder. Only the part that shows up at the wrist and elbow is visible.

## 2. The mapping protocol

The subject sits with the elbow at about 90° and the forearm on the table, thumb up. The other hand or the table
gives resistance. Each task is isometric (no joint movement) at about **half of maximum effort**.

1. Baseline: relax for 8 s.
2. Nine tasks × 3 repetitions in random order. Each repetition: ENTER, 1 s settle, 4 s hold, 3 s rest.

| id | Task | Wrist band label | Upper-arm band label |
|---|---|---|---|
| `rest` | Relax | — | — |
| `wrist_flex` | Wrist flexion against the other hand | flexors (FCR·PL·FCU) | `wrist_tasks` (unclear) |
| `wrist_ext` | Wrist extension | extensors (ECRL·ECRB·ECU) | `wrist_tasks` |
| `radial_dev` | Radial deviation (thumb side up) | radial (ECRL·FCR·BR) | `wrist_tasks` |
| `ulnar_dev` | Ulnar deviation (little-finger side down) | ulnar (FCU·ECU) | `wrist_tasks` |
| `grip` | Squeeze a fist or towel, wrist neutral | finger_flexors (FDS) | `wrist_tasks` |
| `elbow_flex_sup` | Elbow flexion, palm up | brachioradialis | **biceps** |
| `elbow_flex_pro` | Elbow flexion, palm down | brachioradialis | **brachialis** |
| `elbow_ext` | Elbow extension (press down with the fist) | elbow_ext (anconeus etc.) | **triceps** |

Total time is about 5 min. `--map-reps` changes the repetitions (minimum 2).

```bash
emg-probe map --band wrist=/dev/cu.A --band upperarm=/dev/cu.B --note "..."
emg-probe map --simulate --band wrist --band upperarm
emg-probe map --analyse recordings/muscle_map_...        # recompute
```

## 3. What comes out

`muscle_map.json`, `muscle_map.md` (a table) and `muscle_map.png` (a task × channel heatmap), plus raw data per band.

Per channel:

- **muscle / group:** the task that raised the channel's RMS above rest the most.
- **share:** that task's fraction of the channel's total response (selectivity).
- **consistency:** in how many repetitions the same task came first.
- **side:** front/back (upper arm), or palm/dorsal + thumb/little-finger (forearm). Following the sides around the ring
  shows where the box sits.
- **weak:** the response is below max(3 × rest noise, 0.25 × rest RMS). Check contact on that channel.

Per band:

- `baseline`: rest RMS, per channel.
- `reference`: rest plus the largest median task response, per channel.
- `valid`, and `groups` (group → channels).

These are used to normalize later recordings:

```
norm = (RMS − baseline) / (reference − baseline)
```

0 is mapping rest and 1 is as strong as the strongest mapping task. This is not MVC, so values above 1 happen. It
makes muscle groups and scenarios comparable within one donning.

### Placement check

With both bands mapped, `placement` is one of:

| Value | Meaning |
|---|---|
| `ok` | Forearm band responds more to wrist tasks, upper-arm band more to elbow tasks |
| `swapped?` | Both reversed. The ports are probably swapped: exchange the two `--band` ports and restart. |
| `unclear` | Common and harmless. The forearm band also responds strongly to elbow flexion (brachioradialis). |

This check replaces a separate band-role test step.

### Expected result

- **Upper arm:** 2–3 consecutive front channels biceps, 3–4 consecutive back channels triceps, front-lateral brachialis.
- **Forearm:** palm side flexors and finger flexors, dorsal side extensors, thumb side brachioradialis / ECRL.

Surface EMG picks up neighbouring muscles (crosstalk), especially on the forearm. The labels are approximate.
