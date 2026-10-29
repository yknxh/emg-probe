import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emgprobe.bands import Bands, clock_fit, merge_windows, parse_bands
from emgprobe.device import FrameParser, encode
from emgprobe.recorder import WindowRecorder


class DeviceTests(unittest.TestCase):
    def test_encoded_frames_parse_back_through_garbage(self):
        rows = np.array([[i*1000-4000+c for c in range(8)] for i in range(20)])
        data, counter = encode(rows, 250, np.random.default_rng(0))
        parser = FrameParser()
        frames = parser.feed(b'\x00\xd2\x17'+data[:50])+parser.feed(data[50:])
        emg = [f for f in frames if f[0] == 0xAA]
        self.assertEqual([list(v) for _, _, v in emg], rows.tolist())
        self.assertEqual(len(frames), counter-250)
        self.assertTrue(all((b[1]-a[1]) % 256 == 1 for a, b in zip(frames, frames[1:])))
        self.assertEqual(parser.discarded, 3)
        imu = [v for k, _, v in frames if k == 0xBB][0]
        self.assertAlmostEqual(imu[5], 9.80665, delta=.05)


class ParsingTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(parse_bands(['wrist=/dev/a', 'upperarm=/dev/b'], False), {'wrist': '/dev/a', 'upperarm': '/dev/b'})
        self.assertEqual(parse_bands(['wrist', 'upperarm'], True), {'wrist': None, 'upperarm': None})
        for bad in (['elbow=/dev/a'], ['wrist=/dev/a', 'wrist=/dev/b'], ['wrist=/dev/a', 'upperarm=/dev/a'], ['wrist']):
            with self.assertRaises(ValueError):parse_bands(bad, False)


class TimingAndMergeTests(unittest.TestCase):
    def test_clock_fit_recovers_rate(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(0);n = 20000;sample = np.arange(1, n+1)
            # Samples arrive in 40-sample reads that complete 2-12 ms after the last sample; clock 100 ppm fast.
            host = 5+(sample//40*40+40)/(2000*(1+1e-4))+rng.uniform(.002, .012, n)//1e-3*1e-3
            with open(Path(d)/'emg.csv', 'w', newline='') as f:
                w = csv.writer(f);w.writerow(['sample', 'host_time_s', 'segment'])
                w.writerows(zip(sample, host, [1]*n))
            fit = clock_fit(Path(d)/'emg.csv', 2000)
            self.assertAlmostEqual(fit['rate_error_ppm'], 100, delta=30)
            self.assertLess(fit['read_jitter_ms']['p99'], 25)

    def test_merge_nearest_window_in_same_event(self):
        with tempfile.TemporaryDirectory() as d:
            for role, times in [('wrist', [1.0, 1.1, 1.2]), ('upperarm', [1.02, 1.13, 5.0])]:
                (Path(d)/role).mkdir()
                with open(Path(d)/role/'windows.csv', 'w', newline='') as f:
                    w = csv.writer(f);w.writerow(['time_s', 'event_id', 'phase', 'level'])
                    w.writerows([(t, 1 if t < 1.15 else 2, 'free', i) for i, t in enumerate(times)])
            r = merge_windows(d, ['wrist', 'upperarm'], ['time_s', 'event_id', 'phase'])
            with open(Path(d)/'bands_windows.csv') as f:rows = list(csv.DictReader(f))
            self.assertEqual((r['windows'], r['matched']), (3, 2))
            self.assertEqual([x['upperarm_level'] for x in rows], ['0', '1', ''])
            self.assertAlmostEqual(float(rows[1]['upperarm_dt_s']), .03)


class RecordingTests(unittest.TestCase):
    def test_bands_share_clock_and_events(self):
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(port=None, baud=921600, rate=2000, simulate=True)
            bands = Bands(WindowRecorder, Path(d), args, dict(wrist=None, upperarm=None))
            bands.start()
            try:
                deadline = time.monotonic()+4
                while bands.snapshot()['latest'] is None and time.monotonic() < deadline:time.sleep(.05)
                self.assertEqual(len(bands.snapshot()['latest']['rms']), 16)
                bands.set_phase('hold', 1, 1);bands.save_meta(dict(status='recording', kind='test'))
            finally:
                bands.stop()
            events = {role: [json.loads(l) for l in (Path(d)/role/'events.jsonl').read_text().splitlines()] for role in ['wrist', 'upperarm']}
            self.assertEqual(events['wrist'][-1], events['upperarm'][-1])
            self.assertEqual(bands.readers['wrist'].origin, bands.readers['upperarm'].origin)
            m = json.loads((Path(d)/'upperarm'/'metadata.json').read_text())
            self.assertEqual((m['band'], m['kind'], m['status']), ('upperarm', 'test', 'recording'))
            self.assertIn('bands', json.loads((Path(d)/'metadata.json').read_text()))


if __name__ == '__main__':
    unittest.main()
