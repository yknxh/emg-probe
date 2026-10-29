"""Free-form EMG logging with operator-marked contact, one or two bands.

ENTER toggles free <-> contact (contact = hand touching the surface and pressing). Ctrl+C or 'q'+ENTER stops and saves.
--phases approach,grasp,contact,free: a fixed sequence instead; the session starts in the first phase, each ENTER moves
to the next one and ENTER in the last phase ends the session (label = phase index).
Saves 2 kHz raw/filtered EMG, IMU and the 100 ms RMS of every window (features.jsonl, windows.csv).
With --mapping (a muscle map made in the same donning) windows.csv also has the fraction of the mapping response per
channel (norm1-8), its mean over responsive channels (level) and per muscle group (g_<group>).
Contact marks are manual: expect ~0.2-0.3 s reaction delay; no force magnitude is recorded.

usage: emg-probe log --band wrist=/dev/cu.A [--band upperarm=/dev/cu.B] [--mapping recordings/muscle_map_...]
       emg-probe log --simulate --band wrist --band upperarm
batch: add --sessions 20 for 20 separate sessions (contact_batch_.../session_01..20); ENTER starts each, q+ENTER ends the batch early.
recompute: emg-probe log --analyse recordings/contact_log_... [--mapping other muscle map]
-> <session>/<band>/ (single-band layout each) + bands_windows.csv (windows of all bands joined by time)
"""
import argparse
import csv
import json
import select
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emgprobe import DEFAULT_OUTPUT
from emgprobe.analysis import normalize
from emgprobe.bands import Bands, ROLES, merge_windows, parse_bands, session_timing
from emgprobe.device import BAUD
from emgprobe.files import read_jsonl, save_json, session_dir
from emgprobe.filters import filter_settings
from emgprobe.protocols.muscle_map import load_map
from emgprobe.recorder import WindowRecorder
from emgprobe.terminal import band_line, wait_for_data

PHASES = {0: 'free', 1: 'contact'}
RESERVED = ('waiting_calibration', 'end')


class ContactRecorder(WindowRecorder):
    # Every 100 ms window is kept; windows never straddle a phase change.
    FEATURE_PHASES = ('free', 'contact')
    FEATURE_SKIP_S = 0.

    def sim_gain(self, event):
        return np.array([3.]*4+[2.]*4) if event['phase'] == 'contact' else np.ones(8)


def analyse_band(band_dir, phases, band_map=None):
    """windows.csv + per-phase medians from saved RMS windows of one band."""
    p = Path(band_dir)
    out = []
    for r in read_jsonl(p/'features.jsonl'):
        row = dict(time_s=r['end_time_s'], event_id=r['event_id'], phase=r['phase'], contact=int(r['phase'] == 'contact'),
                   contact_index=r['round'], segment=r['segment'], **{f'rms{i+1}': v for i, v in enumerate(r['rms'])})
        if band_map:
            n = normalize(r['rms'], band_map)
            row.update(level=n['level'], **{f'g_{g}': v for g, v in n['groups'].items()}, **{f'norm{i+1}': v for i, v in enumerate(n['norm'])})
        out.append(row)
    if out:
        with (p/'windows.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    keys = ['level']+[k for k in (out[0] if out else {}) if k.startswith('g_')] if band_map else []
    summary = {}
    for phase in phases:
        sel = [r for r in out if r['phase'] == phase]
        summary[phase] = dict(windows=len(sel), median_rms=np.median([[r[f'rms{i+1}'] for i in range(8)] for r in sel], 0).tolist() if sel else None,
                              **{f'median_{k}': float(np.median([r[k] for r in sel])) if sel and all(r[k] is not None for r in sel) else None for k in keys})
    return dict(normalized=bool(band_map), summary=summary)


def analyse_session(directory, mapping=None):
    """Each band, bands_windows.csv and a session analysis.json. `mapping`: muscle map folder; default = the copy saved with the session."""
    p = Path(directory);meta = json.loads((p/'metadata.json').read_text())
    phases = meta.get('phases') or list(PHASES.values())
    source = Path(mapping) if mapping else p/'muscle_map.json' if (p/'muscle_map.json').exists() else None
    bands_map = load_map(source) if source else {}
    roles = list(meta['bands'])
    bands = {role: analyse_band(p/role, phases, bands_map.get(role)) for role in roles}
    merged = merge_windows(p, roles, ['time_s', 'event_id', 'phase', 'contact', 'contact_index']) if len(roles) > 1 else None
    result = dict(mapping_source=str(source.resolve()) if source else None, bands=bands, merged=merged, timing=session_timing(p),
                  note='RMS in filtered counts; level/g_*/norm* = fraction of the muscle-map response (not MVC, not stiffness). '
                       'Contact is operator-marked with reaction delay. bands_windows.csv joins each window of the first band to the '
                       'nearest window of the other band in the same phase (PC clock, not resampled; <band>_dt_s = time offset).')
    save_json(p/'analysis.json', result)
    return result


def fmt(x):
    return ' -- ' if x is None else f'{x:4.2f}'


def run(a, out=None, batch_session=None):
    out = out or session_dir(a.output, 'contact_log')
    out.mkdir(parents=True, exist_ok=True)
    group = Bands(ContactRecorder, out, SimpleNamespace(port=None, baud=a.baud, rate=a.rate, simulate=a.simulate), a.bands)
    phases = a.phases or list(PHASES.values())
    for r in group.readers.values():r.FEATURE_PHASES = tuple(phases)
    if a.mapping:shutil.copy(a.mapping/'muscle_map.json', out/'muscle_map.json')
    meta = dict(kind='contact_log', status='recording', simulated=a.simulate, rate=a.rate, baud=a.baud, filter=filter_settings(a.rate),
                mapping_source=str(a.mapping.resolve()) if a.mapping else None, note=a.note, batch_session=batch_session,
                features='100ms RMS, 100ms stride, all windows (no initial skip)', phases=a.phases,
                label_semantics=(f"label = index in phases {a.phases} (operator ENTER moves to the next phase); round 0; contact column = phase == 'contact'"
                                 if a.phases else 'label 1 = contact (operator ENTER: hand on surface and pressing), 0 = free; round = contact index (0 while free)'),
                timing='PC read-completion monotonic timestamps; manual contact marks lag the true contact by the operator reaction time')
    group.save_meta(meta)
    group.start();status = 'aborted'
    try:
        wait_for_data(group)
        print('Output:', out)
        for role in a.bands:
            if role not in a.bands_map:print(f'{role}: no muscle map, recording channel RMS only.')
        if a.phases:print(f"ENTER: next phase ({' -> '.join(a.phases)} -> end) | q+ENTER or Ctrl+C: stop and save\n", flush=True)
        else:print('ENTER: free <-> contact | q+ENTER or Ctrl+C: stop and save\n', flush=True)
        contact, count, index, start = 0, 0, 0, time.monotonic()
        group.set_phase(phases[0], 0, 0)
        while True:
            while select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line or line.strip().lower() in ['q', 'quit', 'stop']:raise KeyboardInterrupt
                start = time.monotonic()
                if a.phases:
                    index += 1
                    if index == len(a.phases):raise KeyboardInterrupt
                    group.set_phase(a.phases[index], 0, index)
                else:
                    contact = 1-contact;count += contact
                    group.set_phase(PHASES[contact], count if contact else 0, contact)
            s = group.snapshot()
            if s['error']:raise RuntimeError(s['error'])
            stale = s['last_emg'] is None or time.monotonic()-s['last_emg'] > 1
            label = f'{a.phases[index]} {index+1}/{len(a.phases)}' if a.phases else f'CONTACT #{count}' if contact else 'free'
            print(f"\r{label:>11} {time.monotonic()-start:5.1f}s | {band_line(s, a.bands_map)}{' | EMG STALE' if stale else ''}   ", end='', flush=True)
            time.sleep(.1)
    except KeyboardInterrupt:
        status = 'completed'
    except RuntimeError as e:
        meta['error'] = str(e);status = 'error'
    finally:
        group.set_phase('end');group.stop()
        s = group.snapshot()
        meta.update(status='error' if s['error'] else status, error=s['error'] or meta.get('error'), elapsed_s=time.monotonic()-group.primary.origin,
                    aa=s['aa'], bb=s['bb'], counter_breaks=s['breaks'], discarded_bytes=group.discarded())
        group.save_meta(meta)
    if meta['status'] == 'error':
        print('\nerror:', meta['error'], '| saved', out);raise RuntimeError(meta['error'])
    result = analyse_session(out)
    print('\n')
    for role, r in result['bands'].items():
        for phase, v in r['summary'].items():
            print(f"{role} {phase:>7}: {v['windows']} windows | median level {fmt(v.get('median_level'))} | "
                  f"RMS {np.round(v['median_rms']).astype(int).tolist() if v['median_rms'] else []}")
    print('saved', out)


def run_batch(a):
    """Back-to-back sessions in contact_batch_.../session_NN; each reopens the port(s) and is analysed on its own."""
    batch = session_dir(a.output, 'contact_batch')
    done = 0
    for k in range(1, a.sessions+1):
        try:
            line = input(f'\nSession {k}/{a.sessions}: ENTER to start, q+ENTER to end the batch ')
        except (KeyboardInterrupt, EOFError):
            break
        if line.strip().lower() in ['q', 'quit', 'stop']:break
        run(a, batch/f'session_{k:02d}', dict(index=k, planned=a.sessions, batch_dir=str(batch)))
        done = k
    print(f'\n{done}/{a.sessions} sessions saved in', batch)


def main(argv=None):
    p = argparse.ArgumentParser(prog='emg-probe log', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--band', action='append', default=[], metavar='ROLE=PORT', help=f'ROLE in {list(ROLES)} (port optional with --simulate)')
    p.add_argument('--simulate', action='store_true')
    p.add_argument('--baud', type=int, default=BAUD)
    p.add_argument('--rate', type=int, default=2000)
    p.add_argument('--mapping', type=Path, help='Muscle map folder (emg-probe map) made in this donning')
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--note', default='')
    p.add_argument('--sessions', type=int, default=1, help='Record N separate sessions in one contact_batch_ folder')
    p.add_argument('--phases', help='Comma-separated phase sequence, e.g. approach,grasp,contact,free (ENTER advances, ENTER in the last ends)')
    p.add_argument('--analyse', type=Path, help='Recompute windows.csv/analysis.json for a saved session')
    a = p.parse_args(argv)
    if a.analyse:
        print(json.dumps(analyse_session(a.analyse, a.mapping), indent=2));return
    try:
        a.bands = parse_bands(a.band, a.simulate)
        if a.mapping and not (a.mapping/'muscle_map.json').exists():raise ValueError(f'{a.mapping} has no muscle_map.json')
        a.bands_map = load_map(a.mapping, a.rate, a.simulate) if a.mapping else {}
    except ValueError as e:
        p.error(str(e))
    if not a.bands:p.error('Give --band wrist=PORT and/or --band upperarm=PORT')
    if a.sessions < 1:p.error('--sessions must be >= 1')
    a.phases = [x.strip() for x in a.phases.split(',')] if a.phases else None
    if a.phases and (len(a.phases) < 2 or len(set(a.phases)) != len(a.phases) or not all(a.phases) or set(a.phases) & set(RESERVED)):
        p.error(f'--phases needs 2+ distinct names, none of {RESERVED}')
    run_batch(a) if a.sessions > 1 else run(a)


if __name__ == '__main__':
    main()
