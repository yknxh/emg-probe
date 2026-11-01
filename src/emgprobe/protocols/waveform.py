"""Filtered EMG of a saved recording: vertical bars, coloured by size. Display only, the recording is not changed.

usage: python -m emgprobe.protocols.waveform recordings/SESSION[/wrist] [--save out.png]
       python -m emgprobe.protocols.waveform recording.mcap             (needs: pip install mcap mcap-protobuf-support)

Three filters: 60 Hz notches at 60, 120, ..., 540 Hz (mains hum), 20 Hz high-pass (DC offset, slow baseline movement),
450 Hz low-pass (above the surface EMG band). Each bar spans the min..max of 20 ms. All channels share one y scale and
one colour scale. Bars are clipped at the y limit and colour saturates at 8000 counts: raise --scale to see larger bars.
The first 1 s after a gap is not drawn.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import signal

BAR_S, WARMUP_S = .02, 1.
COLOUR_RANGE = (300, 8000)   # counts, log scale: about the 1st and 99th percentile of 20 ms peaks in wiping recordings
COLOURS = ('#111111', '#2a78d6', '#1baf7a', '#eda100', '#ff0000')


def display_sos(rate):
    if rate <= 900:
        raise ValueError('20-450 Hz filtering requires --rate greater than 900 Hz')
    # The 450 Hz edge is gentle (480 Hz still passes at 0.55x), so hum above it needs notches too.
    sections = [signal.butter(4, [20, 450], btype='bandpass', fs=rate, output='sos')]
    for hz in (hz for hz in range(60, 541, 60) if hz < rate/2):
        b, a = signal.iirnotch(hz, hz/4, fs=rate)
        sections += [signal.tf2sos(b, a)]*3
    return np.vstack(sections)


def load_csv(path):
    """[(start time s, raw counts), ...], one item per gap-free stream segment of an emg-probe emg.csv."""
    parts = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, [])
        if 'segment' not in header:
            raise ValueError(f'{path} is empty or not an emg-probe emg.csv')
        time, segment = header.index('host_time_s'), header.index('segment')
        channels = [i for i, name in enumerate(header) if name.startswith('raw_ch')]
        for row in reader:
            if len(row) < len(header):
                break   # interrupted recording: last line cut off
            start, values = parts.setdefault(row[segment], (float(row[time]), []))
            values.append([float(row[i]) for i in channels])
    return [(start, np.array(values)) for start, values in parts.values()]


def load_mcap(path, rate, topic=None):
    """Same as load_csv for a Human-Sensor capture. A jump in first_sample_index marks lost samples."""
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as error:
        raise ImportError(f'{error}; reading .mcap needs: pip install mcap mcap-protobuf-support')
    chunks = {}
    with open(path, 'rb') as f:
        for _, channel, _, message in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
            if channel.topic.startswith('/emg/') and channel.topic.endswith('/emg'):
                chunks.setdefault(channel.topic, []).append(message)
    if not chunks:
        raise ValueError('No /emg/.../emg topic in this file')
    if topic is None and len(chunks) > 1:
        raise ValueError(f'Several EMG topics, pick one with --topic: {sorted(chunks)}')
    if topic is not None and topic not in chunks:
        raise ValueError(f'No topic {topic!r}; found {sorted(chunks)}')
    messages = chunks[topic or next(iter(chunks))]
    parts, expected = [], None
    for m in messages:
        if m.first_sample_index != expected:
            parts.append(((m.first_sample_index-messages[0].first_sample_index)/rate, []))
        parts[-1][1].append(np.array(m.samples, float).reshape(m.num_samples, -1))
        expected = m.first_sample_index+m.num_samples
    return [(start, np.vstack(values)) for start, values in parts]


def bars(parts, rate):
    """Filter each segment from a settled state, then return bar times (s) and per-channel min and max."""
    sos, size = display_sos(rate), round(rate*BAR_S)
    times, low, high = [], [], []
    for start, values in parts:
        zi = signal.sosfilt_zi(sos)[:, :, None]*values[0][None, None, :]
        filtered = signal.sosfilt(sos, values, axis=0, zi=zi)[0]
        first = round(rate*WARMUP_S)
        n = (len(filtered)-first)//size
        if n <= 0:
            continue
        blocks = filtered[first:first+n*size].reshape(n, size, -1)
        times.append(start+(first+size*np.arange(n)+size/2)/rate)
        low.append(blocks.min(1))
        high.append(blocks.max(1))
    if not times:
        raise ValueError(f'Nothing to draw: no segment is longer than the {WARMUP_S:g} s filter warm-up')
    return np.concatenate(times), np.vstack(low), np.vstack(high)


def plot(times, low, high, title, scale):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list('emg_size', COLOURS)
    norm = LogNorm(*COLOUR_RANGE)
    channels = low.shape[1]
    fig, axes = plt.subplots(channels, 1, figsize=(14, 9), sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    for ch, ax in enumerate(axes):
        lines = np.stack([np.c_[times, low[:, ch]], np.c_[times, high[:, ch]]], axis=1)
        collection = LineCollection(lines, array=np.maximum(np.abs(low[:, ch]), np.abs(high[:, ch])), cmap=cmap, norm=norm, linewidths=.7)
        ax.add_collection(collection)
        ax.set_ylabel(f'CH{ch+1}', rotation=0, ha='right', va='center')
        ax.tick_params(labelsize=8)
    axes[0].set_xlim(times[0], times[-1])
    axes[0].set_ylim(-scale, scale)   # fixed, so one spike cannot shrink everything else; larger values are clipped
    axes[0].set_title(f'{title} | 20-450 Hz, 60 Hz notches | y: ±{scale:,g} counts', loc='left', fontsize=10)
    axes[-1].set_xlabel('Time (s)')
    fig.colorbar(collection, ax=list(axes), pad=.01, fraction=.02, label='Size of each bar: 20 ms peak (counts)')
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m emgprobe.protocols.waveform', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('recording', help='band folder with emg.csv, an emg.csv, or an .mcap capture')
    parser.add_argument('--rate', type=int, default=2000, help='sample rate (Hz); the band streams 2 kHz, only simulated sessions differ')
    parser.add_argument('--scale', type=float, default=10000, help='y axis limit (counts)')
    parser.add_argument('--topic', help='EMG topic of an .mcap with several bands')
    parser.add_argument('--save', help='write this image instead of opening a window')
    args = parser.parse_args(argv)
    path = Path(args.recording)
    if path.is_dir():
        path = path/'emg.csv'
    if not path.is_file():
        parser.error(f'{path} not found; for a two-band session give the band folder, e.g. SESSION/wrist')
    try:
        parts = load_mcap(path, args.rate, args.topic) if path.suffix == '.mcap' else load_csv(path)
        times, low, high = bars(parts, args.rate)
    except (ImportError, ValueError) as error:
        parser.error(str(error))
    fig = plot(times, low, high, path.stem if path.suffix == '.mcap' else '/'.join(path.resolve().parent.parts[-2:]), args.scale)
    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
