"""Causal EMG filters shared by live display and saved-session analysis."""
import numpy as np
from scipy import signal


def filter_settings(rate):
    if rate <= 900:
        raise ValueError('20–450 Hz filtering requires --rate greater than 900 Hz')
    # Mains drifts ~0.05 Hz off 60 and hum can exceed EMG by >1000x: cascaded notches on every in-band harmonic.
    # Fixed bandwidth (Q = hz/bandwidth) so upper harmonics do not cut wide slices of the EMG band.
    return dict(version=3, sample_rate_hz=float(rate), bandpass_hz=[20, 450],
                butterworth_order=4, notch_hz=list(range(60, 421, 60)), notch_bandwidth_hz=4, notch_passes=3,
                rms_window_ms=100, warmup_s=1.0, mode='causal, single pass',
                units='counts (not calibrated microvolts)',
                timing='Uniform samples at nominal rate; PC read timestamps are not interpolated',
                gaps='Reset at recorded stream segment, counter discontinuity, or PC read gap > 0.25 s')


class EMGFilter:
    """Keep state across serial chunks; never reset on a protocol stage change."""
    def __init__(self, rate=2000, channels=8):
        self.settings = filter_settings(rate)
        self.rate, self.channels = rate, channels
        sections = [signal.butter(4, [20, 450], btype='bandpass', fs=rate, output='sos')]
        for hz in self.settings['notch_hz']:
            b, a = signal.iirnotch(hz, hz/self.settings['notch_bandwidth_hz'], fs=rate)
            sections += [signal.tf2sos(b, a)]*self.settings['notch_passes']
        self.sos = np.vstack(sections)
        self.window = max(1, round(rate * 0.1))
        self.warmup = max(round(rate), self.window)
        self.reset()

    def reset(self):
        self.zi = None
        self.rms_zi = np.zeros((self.window - 1, self.channels))
        self.seen = 0

    def process(self, values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return values.copy(), values.copy(), np.zeros(0, dtype=bool)
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * values[0][None, None, :]
        filtered, self.zi = signal.sosfilt(self.sos, values, axis=0, zi=self.zi)
        power, self.rms_zi = signal.lfilter(np.ones(self.window)/self.window, [1],
                                          filtered**2, axis=0, zi=self.rms_zi)
        envelope = np.sqrt(np.maximum(power, 0))
        valid = np.arange(self.seen, self.seen+len(values)) >= self.warmup
        self.seen += len(values)
        # Exclude startup ringing from both display and metrics, while retaining raw data.
        filtered[~valid], envelope[~valid] = np.nan, np.nan
        return filtered, envelope, valid


def contiguous_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def process_recording(values, times, rows, rate):
    processor = EMGFilter(rate)
    filtered, rms = np.full(values.shape, np.nan), np.full(values.shape, np.nan)
    valid = np.zeros(len(values), dtype=bool)
    starts = [0]
    for i in range(1, len(rows)):
        step = rows[i].get('counter_step', '')
        segment_changed = rows[i].get('stream_segment') != rows[i-1].get('stream_segment')
        if segment_changed or (step and int(step) != 1) or times[i]-times[i-1] > 0.25:
            starts.append(i)
    for start, end in zip(starts, starts[1:] + [len(values)]):
        if start == end:
            continue
        processor.reset()
        filtered[start:end], rms[start:end], valid[start:end] = processor.process(values[start:end])
    return filtered, rms, valid, starts


def spectrum(values, mask, rate):
    """Welch PSD, only complete 2 s windows within contiguous selected samples."""
    size = round(rate * 2)
    frequency = np.fft.rfftfreq(size, 1/rate)
    total = np.zeros((len(frequency), 8))
    count = 0
    for start, end in contiguous_runs(mask):
        if end-start < size:
            continue
        freq, psd = signal.welch(values[start:end], fs=rate, window='hann', nperseg=size,
                                 noverlap=size//2, axis=0, detrend='constant', scaling='density')
        windows = 1 + (end-start-size)//(size//2)
        total += psd * windows
        count += windows
    return frequency, total/count if count else np.full_like(total, np.nan), int(count)


def spectral_metrics(raw, filtered, mask, rate):
    f, before, windows = spectrum(raw, mask, rate)
    _, after, _ = spectrum(filtered, mask, rate)
    result = []
    for ch in range(8):
        stats = {}
        for hz in (60, 120):
            if not windows:
                stats.update({f'raw_{hz}hz_fraction': None, f'filtered_{hz}hz_fraction': None,
                              f'attenuation_{hz}hz_db': None})
                continue
            band = (f >= hz-1) & (f <= hz+1)
            raw_line, filtered_line = before[band, ch].sum(), after[band, ch].sum()
            raw_total, filtered_total = before[f >= 1, ch].sum(), after[f >= 1, ch].sum()
            stats.update({f'raw_{hz}hz_fraction': float(raw_line/raw_total) if raw_total > 0 else None,
                          f'filtered_{hz}hz_fraction': float(filtered_line/filtered_total) if filtered_total > 0 else None,
                          f'attenuation_{hz}hz_db': float(10*np.log10(raw_line/filtered_line))
                          if raw_line > 0 and filtered_line > 0 else None})
        result.append(stats)
    return result, (f, before, after, windows)
