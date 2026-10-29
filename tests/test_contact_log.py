import csv
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from emgprobe.filters import filter_settings
from emgprobe.protocols.muscle_map import load_map


def fake_map(directory, simulated=True):
    """Muscle map in which every channel rests at RMS 60 (simulated rest) and reaches 180 at the mapping reference."""
    band = dict(baseline=[60.]*8, reference=[180.]*8, valid=[True]*8, groups=dict(flexors=[1, 2, 3, 4], extensors=[5, 6, 7, 8]))
    m = dict(kind='muscle_map', simulated=simulated, filter=filter_settings(2000), bands=dict(wrist=band, upperarm=band, broken=dict(error='x')))
    Path(directory).mkdir(exist_ok=True)
    (Path(directory)/'muscle_map.json').write_text(json.dumps(m))
    return Path(directory)


def run_log(d, extra, inputs):
    cmd = [sys.executable, '-m', 'emgprobe', 'log', '--simulate', '--output', d, '--band', 'wrist', '--band', 'upperarm', *extra]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line, wait in inputs:
        time.sleep(wait);p.stdin.write(line+'\n');p.stdin.flush()
    return p, p.communicate(timeout=60)[0]


class MapLoadingTests(unittest.TestCase):
    def test_simulated_or_mismatched_maps_are_refused_live(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(set(load_map(fake_map(Path(d)/'m'), 2000, True)), {'wrist', 'upperarm'})
            with self.assertRaises(ValueError):load_map(Path(d)/'m', 2000, False)
            with self.assertRaises(ValueError):load_map(Path(d)/'m', 1000, True)
            self.assertIn('wrist', load_map(fake_map(Path(d)/'real', simulated=False), 2000, False))


class ContactTests(unittest.TestCase):
    def test_two_band_contact_session_with_map(self):
        self.two_band_contact_session(True)

    def test_two_band_contact_session_without_map(self):
        self.two_band_contact_session(False)

    def two_band_contact_session(self, with_map):
        with tempfile.TemporaryDirectory() as d:
            extra = ['--mapping', str(fake_map(Path(d)/'map'))] if with_map else []
            p, log = run_log(d, extra, [('', 2.5), ('', 2.), ('', 1.5), ('q', 0)])
            self.assertEqual(p.returncode, 0, log)
            out = next(Path(d).glob('contact_log_*'))
            r = json.loads((out/'analysis.json').read_text())
            for role in ['wrist', 'upperarm']:
                s = r['bands'][role]['summary']
                self.assertGreater(s['contact']['windows'], 10)
                self.assertGreater(np.median(s['contact']['median_rms']), np.median(s['free']['median_rms']))
                if with_map:
                    # Simulated contact triples channels 1-4 (-> norm 1) and doubles 5-8 (-> .5); free stays at rest (0).
                    self.assertAlmostEqual(s['contact']['median_g_flexors'], 1, delta=.2)
                    self.assertAlmostEqual(s['contact']['median_g_extensors'], .5, delta=.2)
                    self.assertLess(abs(s['free']['median_level']), .15)
                else:self.assertNotIn('median_level', s['contact'])
                self.assertEqual(json.loads((out/role/'metadata.json').read_text())['status'], 'completed')
            self.assertEqual((out/'muscle_map.json').exists(), with_map)
            with open(out/'bands_windows.csv') as f:rows = list(csv.DictReader(f))
            self.assertGreater(r['merged']['matched'], .9*len(rows))
            self.assertIn('upperarm_rms8', rows[0]);self.assertEqual('upperarm_level' in rows[0], with_map)
            again = subprocess.run([sys.executable, '-m', 'emgprobe', 'log', '--analyse', str(out)], capture_output=True, text=True)
            self.assertEqual(again.returncode, 0, again.stderr)

    def test_phase_sequence_ends_after_last_phase(self):
        with tempfile.TemporaryDirectory() as d:
            p, log = run_log(d, ['--phases', 'approach,grasp,contact,free'], [('', 2.5), ('', 1.5), ('', 1.5), ('', 2.)])
            self.assertEqual(p.returncode, 0, log)
            out = next(Path(d).glob('contact_log_*'))
            meta = json.loads((out/'metadata.json').read_text())
            self.assertEqual((meta['status'], meta['phases']), ('completed', ['approach', 'grasp', 'contact', 'free']))
            r = json.loads((out/'analysis.json').read_text())
            for role in ['wrist', 'upperarm']:
                s = r['bands'][role]['summary']
                self.assertEqual(list(s), ['approach', 'grasp', 'contact', 'free'])
                self.assertTrue(all(v['windows'] > 5 for v in s.values()))
                events = [json.loads(l) for l in (out/role/'events.jsonl').read_text().splitlines()]
                self.assertEqual([(e['phase'], e['label']) for e in events if e['phase'] in meta['phases']],
                                 [('approach', 0), ('grasp', 1), ('contact', 2), ('free', 3)])
            with open(out/'bands_windows.csv') as f:rows = list(csv.DictReader(f))
            self.assertEqual({x['phase'] for x in rows if x['contact'] == '1'}, {'contact'})


if __name__ == '__main__':
    unittest.main()
