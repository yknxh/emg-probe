import builtins
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib
matplotlib.use('Agg')

from emgprobe.analysis import auc
from emgprobe.protocols import direction
from emgprobe.protocols import muscle_map as mm


def answers(*lines):
    """input() replacement: the given lines, then EOF."""
    it = iter(lines)

    def fake(prompt=''):
        try:return next(it)
        except StopIteration:raise EOFError
    return fake


class PlanTests(unittest.TestCase):
    def test_interleaved_rounds_within_setup(self):
        ids = ['H_light', 'H_stiff_z', 'V_light', 'V_stiff_z']
        plan = direction.make_plan(ids, 3)
        self.assertEqual(len(plan), 12)
        setups = [direction.BY_ID[s['scenario']]['setup'] for s in plan]
        self.assertEqual(setups, ['horizontal']*6+['vertical']*6)
        for k in range(6):self.assertEqual({s['scenario'] for s in plan[2*k:2*k+2]}, set(ids[2*(k//3):2*(k//3)+2]))
        self.assertEqual([s['scenario'] for s in direction.make_plan(ids, 2, 'blocked')][:3], ['H_light', 'H_light', 'H_stiff_z'])

    def test_placement_verdicts(self):
        band = lambda w, e: dict(wrist_task_score=w, elbow_task_score=e)
        self.assertEqual(mm.judge_placement(dict(wrist=band(.8, .3), upperarm=band(.2, .9))), 'ok')
        self.assertEqual(mm.judge_placement(dict(wrist=band(.2, .9), upperarm=band(.8, .3))), 'swapped?')
        self.assertEqual(mm.judge_placement(dict(wrist=band(.8, .3), upperarm=band(.8, .3))), 'unclear')
        self.assertEqual(mm.judge_placement(dict(wrist=band(.8, .3))), 'unknown')

    def test_auc(self):
        self.assertEqual(auc([3, 4], [1, 2]), 1.)
        self.assertEqual(auc([1, 2], [1, 2]), .5)


class SimulatedExperimentTests(unittest.TestCase):
    def test_mapping_then_scenarios_with_redo_note_and_resume(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(mm.MAP_S, baseline=1.5, settle=.1, hold=.9, rest=.1), \
                mock.patch.object(mm, 'TRIM_S', .2), mock.patch.object(direction, 'TASK_TRIM_S', (.3, .2)), mock.patch('sys.stdout'):
            a = SimpleNamespace(simulate=True, rate=2000, baud=921600, map_reps=2, note='test', bands={'wrist': None, 'upperarm': None})
            with mock.patch.object(builtins, 'input', return_value=''):
                result = mm.run_mapping(a, Path(d)/'map')
            self.assertEqual(result['status'], 'completed')
            # The simulation drives channel (3*label + band shift) % 8 for task `label`.
            for role, shift in [('wrist', 0), ('upperarm', 1)]:
                chans = result['bands'][role]['channels']
                for label in range(1, len(mm.TASKS)):
                    c = chans[(3*label+shift) % 8]
                    self.assertTrue(c['valid']);self.assertEqual(c['best_task'], mm.TASK_IDS[label])
                    self.assertEqual(c['group'], mm.MUSCLES[role][mm.TASK_IDS[label]][0])
            self.assertTrue((Path(d)/'map'/'muscle_map.md').exists() and (Path(d)/'map'/'muscle_map.png').exists())

            argv = ['--simulate', '--band', 'wrist', '--band', 'upperarm', '--quiet', '--scenarios', 'H_air,H_light', '--trials', '2',
                    '--pre-s', '.6', '--task-s', '1.6', '--post-s', '.2', '--mapping', str(Path(d)/'map'), '--output', d]
            # trial, note on it, redo it, one more trial, quit with two left.
            with mock.patch.object(builtins, 'input', side_effect=answers('', 'n wet cloth', 'r', '', '', 'q')):
                direction.main(argv)
            exp_dir = next(Path(d).glob('stiffness_direction_*'))
            exp = json.loads((exp_dir/'experiment.json').read_text())
            self.assertEqual(len(exp['entries']), 3)
            self.assertTrue(exp['entries'][0]['discarded']);self.assertEqual(exp['entries'][0]['note'], 'wet cloth')
            self.assertEqual(exp['entries'][1]['index'], 0)
            self.assertEqual(direction.next_index(exp), 2)

            with mock.patch.object(builtins, 'input', side_effect=answers('', '')):
                direction.main(argv[:-4]+['--resume', str(exp_dir)])
            summary = json.loads((exp_dir/'summary.json').read_text())
            self.assertEqual(summary['valid_trials'], 4);self.assertTrue(summary['normalized'])
            with open(exp_dir/'trials.csv') as f:rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            self.assertEqual({r['band'] for r in rows}, {'wrist', 'upperarm'})
            # Scenario label 2 (H_light) drives wrist channel 7 in the simulation; label 1 (H_air) drives channel 4.
            light = [r for r in rows if r['scenario'] == 'H_light' and r['band'] == 'wrist']
            self.assertTrue(all(float(r['norm7']) > .5 > float(r['norm4']) for r in light))
            self.assertTrue(any(c['question'].startswith('수평: 실제 닦기') for c in summary['contrasts']))
            self.assertTrue((exp_dir/'summary.md').exists() and (exp_dir/'summary.png').exists())
            trial = json.loads((exp_dir/exp['entries'][1]['dir']/'wrist'/'metadata.json').read_text())
            self.assertEqual((trial['status'], trial['band'], trial['scenario']['id']), ('completed', 'wrist', exp['plan'][0]['scenario']))


if __name__ == '__main__':
    unittest.main()
