"""Background recorder for one band: raw bytes, filtered EMG, IMU, phase events and RMS feature windows.

Files written to the session (or band) folder:
  raw.bin        every byte read from the port
  emg.csv        per AA sample: PC read time, phase event, counter, stream segment, raw and filtered counts
  imu.csv        per BB frame: gyro (rad/s), accelerometer (m/s^2), gyro with the calibrated bias removed
  events.jsonl   phase marks (event_id, host_time_s, phase, round, label)
  features.jsonl RMS windows of the filtered EMG inside FEATURE_PHASES, after FEATURE_SKIP_S into each phase
Times are PC monotonic read-completion times relative to `origin` (shared by bands recorded together).
"""
import csv
import json
import threading
import time
from collections import deque

import numpy as np
from scipy import signal

from emgprobe.device import FrameParser, encode
from emgprobe.filters import EMGFilter


class Recorder(threading.Thread):
    WINDOW_S, STRIDE_S = .25, .1
    FEATURE_PHASES = ('calibration', 'hold')
    FEATURE_SKIP_S = .75   # windows start this long after a phase change (reaction time)

    def __init__(self, directory, args):
        super().__init__(daemon=True)
        self.directory, self.args = directory, args
        self.lock, self.stop_event = threading.RLock(), threading.Event()
        # Bands recorded together share one origin so their host times are directly comparable.
        self.origin = time.monotonic() if getattr(args, 'origin', None) is None else args.origin
        self.parser, self.filter = FrameParser(), EMGFilter(args.rate)
        self.size, self.stride = round(args.rate*self.WINDOW_S), round(args.rate*self.STRIDE_S)
        self.buffer = deque(maxlen=self.size)
        self.records, self.events = [], []
        self.imu = deque(maxlen=20000)
        self.latest = None
        self.error = None
        self.aa = self.bb = self.breaks = self.segment = 0
        self.prev = self.last_read = self.last_emg = self.last_imu = None
        self.event_seen, self.event_samples, self.until_feature = -1, 0, self.stride
        self.bias = np.zeros(3)
        self.filtered_header = [f'filtered_ch{i+1}' for i in range(8)]
        self.set_phase('waiting_calibration')

    def set_phase(self, phase, round_id=0, label=-1, at=None):
        with self.lock:
            event = dict(event_id=len(self.events), host_time_s=(time.monotonic() if at is None else at)-self.origin,
                         phase=phase, round=round_id, label=label)
            self.events.append(event)
            with (self.directory/'events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(event)+'\n')

    def snapshot(self):
        with self.lock:
            return dict(aa=self.aa, bb=self.bb, breaks=self.breaks, error=self.error,
                        last_emg=self.last_emg, last_imu=self.last_imu, latest=self.latest,
                        imu=self.imu[-1] if self.imu else None)

    def reset_stream(self):
        self.filter.reset()
        self.buffer.clear()
        self.latest = None
        self.until_feature = self.stride
        self.segment += 1

    def sim_gain(self, event):
        """Per-channel noise gain of the simulated EMG in the current phase; protocols override this."""
        return np.ones(8)

    def simulation(self, rng, counter):
        """20 ms of simulated stream: white noise x sim_gain plus 60 Hz mains hum."""
        with self.lock:
            event = self.events[-1].copy()
        n = round(self.args.rate*.02)
        hum = 800*np.sin(2*np.pi*60*(self.aa+np.arange(n))/self.args.rate)
        emg = rng.normal(0, 100, (n, 8))*np.asarray(self.sim_gain(event), float)+hum[:, None]
        return encode(np.rint(emg).astype(int), counter, rng)

    def run(self):
        connection = None
        try:
            if not self.args.simulate:
                import serial
                connection = serial.Serial(self.args.port, self.args.baud, timeout=.05)
            with (self.directory/'raw.bin').open('wb') as raw, (self.directory/'emg.csv').open('w', newline='') as ef, \
                    (self.directory/'imu.csv').open('w', newline='') as imf, (self.directory/'features.jsonl').open('w') as ff:
                ew, iw = csv.writer(ef), csv.writer(imf)
                ew.writerow(['sample', 'host_time_s', 'event_id', 'phase', 'round', 'label', 'counter', 'segment', 'valid',
                             *[f'raw_ch{i+1}' for i in range(8)], *self.filtered_header])
                iw.writerow(['host_time_s', 'event_id', 'phase', 'round', 'label', 'counter', 'gx_rad_s', 'gy_rad_s', 'gz_rad_s',
                             'ax_m_s2', 'ay_m_s2', 'az_m_s2', 'corrected_gx', 'corrected_gy', 'corrected_gz'])
                rng, counter, next_read, last_flush = np.random.default_rng(47), 0, time.monotonic(), time.monotonic()
                while not self.stop_event.is_set():
                    if self.args.simulate:
                        self.stop_event.wait(max(0, next_read-time.monotonic()))
                        if self.stop_event.is_set():break
                        next_read += .02
                        chunk, counter = self.simulation(rng, counter)
                    else:
                        chunk = connection.read(min(max(connection.in_waiting, 1), 8192))
                    if not chunk:continue
                    now = time.monotonic()
                    raw.write(chunk)
                    with self.lock:
                        event = self.events[-1].copy()
                        if self.last_read is not None and now-self.last_read > .25:
                            self.reset_stream()
                        self.last_read = now
                        frames = self.parser.feed(chunk)
                        # Filter contiguous AA batches, splitting on any shared-counter gap.
                        batch = []

                        def flush():
                            if batch:
                                self.consume_emg(batch, event, now, ew, ff)
                                batch.clear()
                        for kind, seq, values in frames:
                            if self.prev is not None and (seq-self.prev) % 256 != 1:
                                flush()
                                self.breaks += 1
                                self.reset_stream()
                            self.prev = seq
                            if kind == 0xAA:
                                batch.append((seq, values))
                            else:
                                self.bb += 1
                                self.last_imu = now
                                self.imu.append((now-self.origin, event['event_id'], values))
                                iw.writerow([now-self.origin, event['event_id'], event['phase'], event['round'], event['label'], seq,
                                             *values, *(np.array(values[:3])-self.bias)])
                        flush()
                    if now-last_flush > 1:
                        for f in (raw, ef, imf, ff):f.flush()
                        last_flush = now
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
        finally:
            if connection is not None:connection.close()

    def consume_emg(self, batch, event, now, writer, feature_file):
        values = np.array([b[1] for b in batch], float)
        filtered, _, valid = self.filter.process(values)
        if self.event_seen != event['event_id']:
            self.event_seen = event['event_id']
            self.event_samples = 0
            self.buffer.clear()
            self.until_feature = self.stride
        for (seq, raw), value, ok in zip(batch, filtered, valid):
            self.aa += 1
            self.last_emg = now
            self.event_samples += 1
            writer.writerow([self.aa, now-self.origin, event['event_id'], event['phase'], event['round'], event['label'], seq,
                             self.segment, int(ok), *raw, *[float(v) if ok else '' for v in value]])
            if not ok:
                self.buffer.clear()
                continue
            self.buffer.append(value)
            self.until_feature -= 1
            # Only complete windows from the same phase event.
            if len(self.buffer) == self.size and self.until_feature <= 0:
                self.until_feature = self.stride
                window = np.array(self.buffer)
                rms = np.sqrt(np.mean(window**2, axis=0))
                record = dict(**event, end_time_s=now-self.origin, feature=np.log(np.maximum(rms, 1e-6)).tolist(), rms=rms.tolist(), segment=self.segment)
                self.latest = record
                if event['phase'] in self.FEATURE_PHASES and self.event_samples >= round(self.args.rate*self.FEATURE_SKIP_S):
                    # Harmonic concentration is a quality indicator, not an EMG SNR.
                    freq, power = signal.periodogram(window, fs=self.args.rate, axis=0, window='hann')
                    band = (freq >= 20) & (freq <= 450)
                    harmonic = np.zeros(len(freq), bool)
                    for hz in range(60, 421, 60):harmonic |= np.abs(freq-hz) <= 4
                    record['harmonic_fraction'] = (power[harmonic].sum(0)/np.maximum(power[band].sum(0), 1e-20)).tolist()
                    self.records.append(record)
                    feature_file.write(json.dumps(record)+'\n')


class WindowRecorder(Recorder):
    """100 ms windows with a 100 ms stride (non-overlapping), as used by the terminal protocols."""
    WINDOW_S = .1
