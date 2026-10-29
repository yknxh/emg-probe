import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from emgprobe import cli
from emgprobe.protocols import check


class CliTests(unittest.TestCase):
    def test_help_lists_every_command(self):
        out = io.StringIO()
        with redirect_stdout(out):cli.main([])
        for name in cli.COMMANDS:self.assertIn(name, out.getvalue())
        with self.assertRaises(SystemExit):cli.main(['nope'])

    def test_simulated_signal_check(self):
        with tempfile.TemporaryDirectory() as d:
            with redirect_stdout(io.StringIO()):channels = cli.main(['check', '--simulate', '--seconds', '1.6', '--output', d])
            self.assertTrue(all(c['ratio'] > 2 for c in channels))
            # The simulated stream carries strong 60 Hz hum in the raw signal.
            self.assertTrue(all(c['raw_60hz_share_at_rest'] > .5 for c in channels))
            out = next(Path(d).glob('signal_check_*'))
            self.assertEqual(len(json.loads((out/'analysis.json').read_text())['channels']), 8)
        self.assertEqual(check.STEPS[1][0], 'hold')


if __name__ == '__main__':
    unittest.main()
