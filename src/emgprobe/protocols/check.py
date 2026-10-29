"""30 s signal check of one band: 10 s rest, 10 s strong fist, 10 s rest.

Prints per channel the rest and fist RMS (100 ms windows, first 0.75 s of each phase skipped), the fist/rest ratio
(>= 2: OK) and the share of 60 Hz harmonics in the raw signal at rest (mains pickup / poor electrode contact).

usage: emg-probe check [--port /dev/cu.A] [--simulate]
"""
import argparse
import csv
import time
from types import SimpleNamespace

import numpy as np

from emgprobe import DEFAULT_OUTPUT
from emgprobe.device import BAUD, find_port
from emgprobe.files import save_json, session_dir
from emgprobe.filters import filter_settings
from emgprobe.recorder import WindowRecorder

SEC = 10
STEPS = [('rest', 0, '힘 빼고 쉬기'), ('hold', 1, '주먹 꽉 쥐기'), ('rest', 0, '다시 힘 빼기')]


class CheckRecorder(WindowRecorder):
    FEATURE_PHASES = ('rest', 'hold')

    def sim_gain(self, event):
        return np.full(8, 4.) if event['phase'] == 'hold' else np.ones(8)


def line_share(raw, rate):
    """Fraction of 20-450 Hz power within +-1 Hz of the 60 Hz harmonics."""
    x = raw-raw.mean()
    f = np.fft.rfftfreq(len(x), 1/rate)
    p = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
    band = (f > 20) & (f < 450)
    line = sum(((f > k*60-1) & (f < k*60+1)) for k in range(1, 8)).astype(bool)
    return float(p[band & line].sum()/p[band].sum())


def raw_of_event(emg_csv, event_id):
    with open(emg_csv) as f:
        r = csv.DictReader(f)
        return np.array([[float(row[f'raw_ch{c+1}']) for c in range(8)] for row in r if int(row['event_id']) == event_id])


def main(argv=None):
    p = argparse.ArgumentParser(prog='emg-probe check', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port')
    p.add_argument('--simulate', action='store_true')
    p.add_argument('--baud', type=int, default=BAUD)
    p.add_argument('--rate', type=int, default=2000)
    p.add_argument('--seconds', type=float, default=SEC, help='Length of each of the three phases')
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    a = p.parse_args(argv)
    if not a.port and not a.simulate:
        try:a.port = find_port()
        except ValueError as e:p.error(str(e))
    directory = session_dir(a.output, 'signal_check')
    reader = CheckRecorder(directory, SimpleNamespace(port=a.port, baud=a.baud, rate=a.rate, simulate=a.simulate))
    reader.start()
    try:
        deadline = time.monotonic()+5
        while reader.snapshot()['latest'] is None:
            if time.monotonic() > deadline or reader.error:
                raise RuntimeError(reader.error or 'EMG startup timeout')
            time.sleep(.05)
        for phase, label, cue in STEPS:
            reader.set_phase(phase, label=label)
            start = time.monotonic()
            while (left := a.seconds-(time.monotonic()-start)) > 0:
                ss = reader.snapshot()
                if ss['error'] or time.monotonic()-ss['last_emg'] > 1:
                    raise RuntimeError(ss['error'] or 'EMG stale')
                rms = ss['latest']['rms'] if ss['latest'] else []
                print(f"{cue}: {left:4.1f}s | RMS {np.round(rms).astype(int).tolist()}", end='\r', flush=True)
                time.sleep(.1)
            print()
    finally:
        reader.set_phase('end')
        reader.stop_event.set()
        reader.join()
    with reader.lock:records = list(reader.records)
    rest = np.median([r['rms'] for r in records if r['phase'] == 'rest'], 0)
    fist = np.median([r['rms'] for r in records if r['phase'] == 'hold'], 0)
    first_rest = next(e['event_id'] for e in reader.events if e['phase'] == 'rest')
    raw = raw_of_event(directory/'emg.csv', first_rest)
    channels = [dict(channel=c+1, rest_rms=float(rest[c]), fist_rms=float(fist[c]), ratio=float(fist[c]/rest[c]),
                     raw_60hz_share_at_rest=line_share(raw[:, c], a.rate)) for c in range(8)]
    print('\nch | rest RMS | fist RMS | fist/rest | raw 60Hz share at rest')
    for c in channels:
        print(f"{c['channel']:2d} | {c['rest_rms']:8.0f} | {c['fist_rms']:8.0f} | {c['ratio']:9.1f} | {c['raw_60hz_share_at_rest']:5.2f}  "
              f"{'OK' if c['ratio'] >= 2 else 'weak'}")
    save_json(directory/'metadata.json', dict(kind='signal_check', simulated=a.simulate, port=a.port, rate=a.rate, filter=filter_settings(a.rate),
                                              seconds=a.seconds, aa=reader.aa, bb=reader.bb, counter_breaks=reader.breaks, error=reader.error))
    save_json(directory/'analysis.json', dict(channels=channels, note='fist/rest >= 2 is a rough contact check, not a calibration'))
    print('saved', directory)
    return channels


if __name__ == '__main__':
    main()
