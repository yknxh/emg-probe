import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emgprobe.protocols.effort import EffortApp, analyse

class EffortTests(unittest.TestCase):
    def test_collection_and_partial_save(self):
        with tempfile.TemporaryDirectory() as d:
            a=EffortApp(SimpleNamespace(output=d,port=None,baud=921600,rate=2000,rounds=2,note='simulation',simulate=True))
            a.fig.canvas.draw_idle=lambda:None
            a.prepare_s=a.rest_s=.02;a.calibration_s=1.3
            for step in a.plan:step['seconds']=1.05
            try:
                deadline=time.monotonic()+35
                while not a.finished and time.monotonic()<deadline:
                    if a.phase.startswith('waiting'):
                        index=a.index;a.tick();self.assertEqual(index,a.index);a.ready()
                    time.sleep(.02);a.tick()
                self.assertTrue(a.finished)
                self.assertEqual(a.meta['status'],'completed')
                result=json.loads((a.directory/'analysis.json').read_text())
                self.assertEqual(len(result['trials']),10)
                self.assertEqual([r['repetitions'] for r in result['levels']],[2]*4)
                self.assertIsNone(result['measured_force_N'])
                self.assertIsNone(result['measured_stiffness'])
                self.assertTrue(all(all(r['strictly_increasing_channels']) for r in result['ordering_by_round']))
                self.assertEqual((a.directory/'raw.bin').stat().st_size,29*(a.meta['aa']+a.meta['bb']))
                files=['emg.csv','imu.csv','features.jsonl','raw.bin','events.jsonl','metadata.json']
                hashes=lambda:[hashlib.sha256((a.directory/p).read_bytes()).hexdigest() for p in files]
                before=hashes();analyse(a.directory);self.assertEqual(before,hashes())
            finally:a.finish('aborted');plt.close(a.fig)
            b=EffortApp(SimpleNamespace(output=d,port=None,baud=921600,rate=2000,rounds=2,note='',simulate=True))
            b.fig.canvas.draw_idle=lambda:None
            try:
                b.finish('aborted')
                result=json.loads((b.directory/'analysis.json').read_text())
                self.assertEqual(result['trials'],[])
                self.assertTrue(all(x['repetitions']==0 for x in result['levels']))
            finally:b.finish('aborted');plt.close(b.fig)

if __name__=='__main__':unittest.main()
