"""Offline helpers on saved sessions: phase windows, muscle-map normalization, IMU motion, AUC."""
import csv
from pathlib import Path

import numpy as np

from emgprobe.files import read_jsonl


def phase_windows(band_dir, phase, trim_s, end_trim_s=0.):
    """{event_id: [feature record, ...]} of `phase`, without windows ending < trim_s after the phase start or < end_trim_s before its end."""
    events = read_jsonl(Path(band_dir)/'events.jsonl')
    starts = {e['event_id']: e['host_time_s'] for e in events}
    ends = {e['event_id']: n['host_time_s'] for e, n in zip(events, events[1:])}
    out = {}
    for r in read_jsonl(Path(band_dir)/'features.jsonl'):
        ev = r['event_id']
        if r['phase'] != phase or r['end_time_s']-starts[ev] < trim_s:continue
        if end_trim_s and ev in ends and ends[ev]-r['end_time_s'] < end_trim_s:continue
        out.setdefault(ev, []).append(r)
    return out


def normalize(rms, band):
    """RMS as a fraction of one band's muscle-map response: 0 = mapping rest, 1 = strongest mapping task (not MVC).
    Returns per-channel norm (None for unresponsive channels), level (mean over responsive channels) and per-group means."""
    b, ref, ok = np.array(band['baseline']), np.array(band['reference']), np.array(band['valid'])
    with np.errstate(divide='ignore', invalid='ignore'):norm = (np.asarray(rms, float)-b)/(ref-b)
    return dict(norm=[float(x) if v else None for x, v in zip(norm, ok)], level=float(np.mean(norm[ok])) if ok.any() else None,
                groups={g: float(np.mean([norm[c-1] for c in chs])) for g, chs in band['groups'].items()})


def level(rms, band):
    """Mean fraction of the mapping response over responsive channels, or None without a usable map."""
    return normalize(rms, band)['level'] if band else None


def task_imu(band_dir, events, t0, t1):
    """Gyro RMS around the median (rad/s) and mean acceleration of the given events between PC times t0 and t1."""
    g, acc = [], []
    with open(Path(band_dir)/'imu.csv') as f:
        for r in csv.DictReader(f):
            if int(r['event_id']) in events and t0 <= float(r['host_time_s']) <= t1:
                g.append([float(r[k]) for k in ('gx_rad_s', 'gy_rad_s', 'gz_rad_s')]);acc.append([float(r[k]) for k in ('ax_m_s2', 'ay_m_s2', 'az_m_s2')])
    if len(g) < 10:return None
    g = np.array(g)
    return dict(gyro_rms_rad_s=float(np.sqrt(np.mean(np.sum((g-np.median(g, 0))**2, 1)))), acc_mean_m_s2=np.mean(acc, 0).tolist())


def auc(x, y):
    """P(a value of x > a value of y), ties count half."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(((x[:, None] > y[None]).sum()+.5*(x[:, None] == y[None]).sum())/(len(x)*len(y)))
