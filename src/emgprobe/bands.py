"""Several identical EMG bands recorded side by side on one PC.

Each band keeps the single-band file layout in its own sub-folder (<session>/<role>/emg.csv, imu.csv, events.jsonl, ...),
so every per-band analysis runs unchanged. Bands have independent sample clocks and 8-bit counters: the only shared
time base is the PC monotonic clock (read-completion times, ~10-30 ms jitter). All recorders share one origin and every
phase mark is written to every band with the same timestamp, so event_id and host_time_s line up across bands.
"""
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emgprobe.device import port_info
from emgprobe.files import save_json

ROLES = ('wrist', 'upperarm')


def parse_bands(items, simulate):
    """['wrist=/dev/cu.A', 'upperarm=/dev/cu.B'] -> {'wrist': '/dev/cu.A', ...}; the port may be omitted when simulating."""
    bands = {}
    for item in items:
        role, _, port = item.partition('=')
        if role not in ROLES:raise ValueError(f'Unknown band role {role!r}; use {list(ROLES)}')
        if role in bands:raise ValueError(f'Band {role} given twice')
        if not port and not simulate:raise ValueError(f'--band {role}=PORT needs a port')
        bands[role] = port or None
    ports = [p for p in bands.values() if p]
    if len(set(ports)) != len(ports):raise ValueError('Each band needs its own port')
    return bands


class Bands:
    """Recorders of one session. Without bands, a single recorder writes to the session folder itself."""

    def __init__(self, recorder_class, directory, args, bands=None):
        self.directory, self.multi, self.simulate = Path(directory), bands is not None, args.simulate
        origin = time.monotonic()
        self.readers, self.info = {}, {}
        for role, port in (bands.items() if self.multi else [(None, args.port)]):
            d = self.directory/role if role else self.directory
            d.mkdir(parents=True, exist_ok=True)
            self.readers[role] = recorder_class(d, SimpleNamespace(**{**vars(args), 'port': port, 'origin': origin}))
            if role:self.info[role] = dict(port=port, usb=None if args.simulate else port_info(port))
        self.primary = next(iter(self.readers.values()))

    def dirs(self):
        return {role: r.directory for role, r in self.readers.items()}

    def start(self):
        for r in self.readers.values():r.start()

    def stop(self):
        for r in self.readers.values():r.stop_event.set()
        for r in self.readers.values():
            if r.ident is not None:r.join()

    def set_phase(self, phase, round_id=0, label=-1):
        at = time.monotonic()
        for r in self.readers.values():r.set_phase(phase, round_id, label, at=at)

    def snapshot(self):
        """Single-band snapshot, or all bands folded into one: the stalest timestamps, summed counts, rms concatenated in band order."""
        if not self.multi:return self.primary.snapshot()
        snaps = {role: r.snapshot() for role, r in self.readers.items()}
        oldest = lambda k: None if any(s[k] is None for s in snaps.values()) else min(s[k] for s in snaps.values())
        latest = None
        if all(s['latest'] for s in snaps.values()):
            latest = dict(snaps[next(iter(snaps))]['latest'], end_time_s=min(s['latest']['end_time_s'] for s in snaps.values()),
                          rms=[v for s in snaps.values() for v in s['latest']['rms']])
        errors = [f'{role}: {s["error"]}' for role, s in snaps.items() if s['error']]
        return dict(aa=sum(s['aa'] for s in snaps.values()), bb=sum(s['bb'] for s in snaps.values()),
                    breaks=sum(s['breaks'] for s in snaps.values()), error='; '.join(errors) or None,
                    last_emg=oldest('last_emg'), last_imu=oldest('last_imu'), latest=latest, bands=snaps)

    def discarded(self):
        return sum(r.parser.discarded for r in self.readers.values())

    def save_meta(self, meta):
        """Session metadata; with bands, also one metadata.json per band that reads like a single-band session."""
        if self.multi:
            for role, r in self.readers.items():
                s = r.snapshot()
                self.info[role].update(aa=s['aa'], bb=s['bb'], counter_breaks=s['breaks'], discarded_bytes=r.parser.discarded, error=s['error'])
            meta['bands'] = self.info
        save_json(self.directory/'metadata.json', meta)
        if not self.multi:return
        for role, r in self.readers.items():
            m = {k: v for k, v in meta.items() if k != 'bands'}
            m.update(self.info[role], band=role, error=self.info[role]['error'] or meta.get('error'))
            if isinstance(meta.get('calibration'), dict) and role in meta['calibration']:m['calibration'] = meta['calibration'][role]
            save_json(r.directory/'metadata.json', m)


def clock_fit(emg_csv, rate):
    """host_time ~= t0_s + sample/rate_hz on the longest gap-free segment: sample-clock error and PC read jitter of one band."""
    rows = {}
    with open(emg_csv) as f:
        r = csv.reader(f);h = next(r);i, j, k = h.index('sample'), h.index('host_time_s'), h.index('segment')
        for row in r:rows.setdefault(row[k], []).append((float(row[i]), float(row[j])))
    if not rows:return None
    seg, xy = max(rows.items(), key=lambda kv: len(kv[1]))
    x, y = np.array(xy).T
    if len(x) < rate:return None
    slope, t0 = np.polyfit(x, y, 1);res = (y-(t0+slope*x))*1e3
    return dict(segment=int(seg), samples=len(x), t0_s=float(t0), rate_hz=float(1/slope), rate_error_ppm=float((1/slope/rate-1)*1e6),
                read_jitter_ms=dict(p50=float(np.percentile(abs(res), 50)), p99=float(np.percentile(abs(res), 99))))


def merge_windows(directory, roles, shared, name='windows.csv', out='bands_windows.csv', tolerance_s=.06):
    """One row per window of the first band, joined to the nearest window of each other band in the same phase event.
    Columns in `shared` come from the first band; the rest are prefixed with the band role. Bands are not resampled."""
    tables = {}
    for role in roles:
        with open(Path(directory)/role/name) as f:tables[role] = list(csv.DictReader(f))
    base, rows = roles[0], []
    by_event = {role: {} for role in roles[1:]}
    for role in roles[1:]:
        for r in tables[role]:by_event[role].setdefault(r['event_id'], []).append(r)
    for b in tables[base]:
        row = {k: b[k] for k in shared}
        for role in roles:
            if role == base:
                match = b
            else:
                match = min(by_event[role].get(b['event_id'], []), key=lambda r: abs(float(r['time_s'])-float(b['time_s'])), default=None)
                if match is not None and abs(float(match['time_s'])-float(b['time_s'])) > tolerance_s:match = None
                row[f'{role}_dt_s'] = '' if match is None else float(match['time_s'])-float(b['time_s'])
            for k in (tables[role][0] if tables[role] else []):
                if k not in shared:row[f'{role}_{k}'] = '' if match is None else match[k]
        rows.append(row)
    if rows:
        with open(Path(directory)/out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return dict(file=out, windows=len(rows), matched=int(sum(all(r.get(f'{x}_dt_s') != '' for x in roles[1:]) for r in rows)),
                tolerance_s=tolerance_s, base_band=base)


def session_timing(directory):
    """clock_fit of every band of a saved session."""
    meta = json.loads((Path(directory)/'metadata.json').read_text())
    return {role: clock_fit(Path(directory)/role/'emg.csv', meta.get('rate', 2000)) for role in meta['bands']}
