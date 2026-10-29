"""Terminal-UI helpers for the guided protocols: startup wait, timed phases with a live line, sounds, prompts."""
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from emgprobe.analysis import level


def beep(quiet, sound='Tink'):
    """macOS system sound (afplay), else the terminal bell."""
    if quiet:return
    path = Path(f'/System/Library/Sounds/{sound}.aiff')
    if shutil.which('afplay') and path.exists():subprocess.Popen(['afplay', str(path)])
    else:print('\a', end='', flush=True)


def wait_for_data(group, timeout_s=5.):
    deadline = time.monotonic()+timeout_s
    while group.snapshot()['latest'] is None:
        s = group.snapshot()
        if time.monotonic() > deadline or s['error']:raise RuntimeError(s['error'] or 'EMG startup timeout')
        time.sleep(.05)


def band_line(snapshot, bands_map=None):
    """'wrist 0.42 | upperarm RMS 310': mapping level per band, or median RMS without a map."""
    parts = []
    for role, b in snapshot['bands'].items():
        if b['latest'] is None:parts.append(f'{role} --');continue
        x = level(b['latest']['rms'], (bands_map or {}).get(role))
        parts.append(f'{role} {x:4.2f}' if x is not None else f"{role} RMS {np.median(b['latest']['rms']):5.0f}")
    return ' | '.join(parts)


def timed(group, phase, round_id, label, seconds, text, bands_map=None, pace_s=0., quiet=True):
    """Hold one phase for `seconds` with a live line; raises RuntimeError on a band error or stale EMG."""
    group.set_phase(phase, round_id, label);start = time.monotonic();tick = pace_s
    while (elapsed := time.monotonic()-start) < seconds:
        s = group.snapshot()
        if s['error'] or s['last_emg'] is None or time.monotonic()-s['last_emg'] > 1:raise RuntimeError(s['error'] or 'EMG stale')
        print(f"\r{text} {seconds-elapsed:4.1f}s | {band_line(s, bands_map)}   ", end='', flush=True)
        if pace_s and elapsed >= tick:beep(quiet, 'Pop');tick += pace_s
        time.sleep(.05)
    print()


def ask(prompt):
    """input() that maps Ctrl+C / EOF to 'q'."""
    try:return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):return 'q'
