import json
import csv
import io
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from emgprobe.protocols.gesture import GestureApp, fit_model, evaluate, predict, load_model
from emgprobe.recorder import Recorder


class GestureTests(unittest.TestCase):
    def test_gap_discards_windows_and_rewarms_filter(self):
        with tempfile.TemporaryDirectory() as d:
            reader=Recorder(Path(d),SimpleNamespace(rate=2000))
            reader.set_phase('hold',1,1)
            event=reader.events[-1]
            values=np.random.default_rng(8).normal(0,100,(3000,8))
            writer=csv.writer(io.StringIO())
            features=io.StringIO()
            reader.consume_emg([(i%256,v) for i,v in enumerate(values)],event,time.monotonic(),writer,features)
            self.assertIsNotNone(reader.latest)
            count=len(reader.records)
            reader.reset_stream()
            reader.consume_emg([(i%256,v) for i,v in enumerate(values[:500])],event,time.monotonic(),writer,features)
            self.assertIsNone(reader.latest)
            self.assertEqual(len(reader.records),count)
            self.assertEqual(reader.segment,1)

    def test_held_out_data_cannot_change_fitted_model(self):
        rng=np.random.default_rng(81)
        y=np.repeat(np.arange(3),60)
        centers=np.array([[0]*8,[2,2,2,0,0,0,1,1],[0,0,0,1,2,2,2,1]])
        x=centers[y]+rng.normal(0,.12,(len(y),8))
        model=fit_model(x,y)
        original=json.dumps(model,sort_keys=True)
        records=[dict(feature=v.tolist(),label=int(k),round=7,phase='hold',event_id=int(k)) for v,k in zip(x,y)]
        good=evaluate(records,model,[7])
        self.assertEqual(good['trial_accuracy'],1.)
        for r in records:r['label']=(r['label']+1)%3
        bad=evaluate(records,model,[7])
        self.assertEqual(bad['trial_accuracy'],0.)
        self.assertEqual(original,json.dumps(model,sort_keys=True))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.json'
            p.write_text(json.dumps(model))
            copy=load_model(p,2000)
            np.testing.assert_array_equal(predict(copy,x)[0],predict(model,x)[0])
            with self.assertRaises(ValueError):load_model(p,1000)

    def test_full_guided_recording_and_frozen_model(self):
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(output=d,port=None,baud=921600,rate=2000,rounds=4,model=None,simulate=True)
            app=GestureApp(args)
            app.fig.canvas.draw_idle=lambda:None
            app.durations.update(calibration=1.3,settle=.025,hold=1.,rest=.025)
            try:
                deadline=time.monotonic()+45
                while app.phase!='live' and not app.finished and time.monotonic()<deadline:
                    if app.phase.startswith('waiting'):
                        index=app.index
                        app.tick()
                        self.assertEqual(index,app.index)
                        app.ready()
                    time.sleep(.02)
                    app.tick()
                self.assertEqual(app.phase,'live',app.meta)
                self.assertTrue(app.meta['calibration']['applied'])
                np.testing.assert_allclose(app.meta['calibration']['gyro_bias_candidate_rad_s'],[.01,-.008,-.006],atol=.002)
                result=json.loads((app.directory/'analysis.json').read_text())
                self.assertEqual(result['train_rounds'],[1,2])
                self.assertEqual(result['test_rounds'],[3,4])
                self.assertEqual(len(result['trials']),6)
                self.assertGreater(result['trial_accuracy'],.8)
                train=[r for r in app.reader.records if r['phase']=='hold' and r['round'] in [1,2]]
                reference=fit_model([r['feature'] for r in train],[r['label'] for r in train])
                for key in ['center','scale','weights','intercept']:
                    np.testing.assert_allclose(app.model[key],reference[key])
                self.assertEqual(app.reader.breaks,0)
                self.assertTrue(all(r['end_time_s']-r['host_time_s']>.6 for r in app.reader.records))
                app.finish('aborted')
                self.assertEqual(app.meta['status'],'completed')
                model_path=app.directory/'model.json'
                raw_size=(app.directory/'raw.bin').stat().st_size
                self.assertEqual(raw_size,29*(app.meta['aa']+app.meta['bb']))
                args.model=model_path
                second=GestureApp(args)
                second.fig.canvas.draw_idle=lambda:None
                try:
                    self.assertEqual(second.model,app.model)
                    second.finish('aborted')
                    self.assertNotIn('evaluation_complete',second.meta)
                    self.assertFalse((second.directory/'analysis.json').exists())
                finally:
                    second.finish('aborted')
                    plt.close(second.fig)
            finally:
                app.finish('aborted')
                plt.close(app.fig)


if __name__=='__main__':unittest.main()
