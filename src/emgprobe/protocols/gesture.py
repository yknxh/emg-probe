"""Guided EMG gesture collection, held-out evaluation and live prediction.

usage: emg-probe gesture [--port /dev/cu.A] [--simulate]
Reuse a frozen model after re-donning: --model recordings/gesture_.../model.json
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from emgprobe import DEFAULT_OUTPUT
from emgprobe.device import BAUD, find_port
from emgprobe.files import save_json, session_dir
from emgprobe.filters import filter_settings
from emgprobe.recorder import Recorder

LABELS = ['rest', 'fist', 'open']
KOREAN = ['휴식: 손과 손가락의 힘을 빼세요', '주먹: 편한 힘으로 주먹을 쥐세요',
          '펴기: 손가락을 펴서 유지하세요 (과하게 힘주지 마세요)']


def make_plan(rounds, seed=2026):
    rng = np.random.default_rng(seed)
    return [(r, int(label)) for r in range(1, rounds+1) for label in rng.permutation(3)]


def fit_model(features, labels, rate=2000):
    """Training-only standardization and regularized shared-covariance LDA."""
    x, y = np.asarray(features, float), np.asarray(labels, int)
    if any(np.sum(y == k) < 2 for k in range(3)):
        raise ValueError('각 동작에 유효 학습 구간이 2개 이상 필요합니다.')
    center, scale = x.mean(0), np.maximum(x.std(0), .05)
    z = (x-center)/scale
    means = np.array([z[y == k].mean(0) for k in range(3)])
    residual = z-means[y]
    covariance = residual.T @ residual / max(1, len(x)-3)
    covariance = .8*covariance + .2*np.eye(8)*max(np.trace(covariance)/8, .01)
    weights = np.linalg.solve(covariance, means.T).T
    return dict(version=1, labels=LABELS, feature='log RMS of 250ms filtered EMG, stride 100ms',
                rate=rate, filter=filter_settings(rate), center=center.tolist(), scale=scale.tolist(),
                weights=weights.tolist(), intercept=(-.5*np.sum(means*weights, axis=1)).tolist())


def predict(model, features):
    z = (np.asarray(features)-model['center'])/model['scale']
    scores = z @ np.array(model['weights']).T + model['intercept']
    p = np.exp(scores-np.max(scores, axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return np.argmax(p, axis=-1), p


def load_model(path, rate):
    model = json.loads(Path(path).read_text(encoding='utf-8'))
    if model.get('version') != 1 or model.get('labels') != LABELS or model.get('rate') != rate or model.get('filter') != filter_settings(rate):
        raise ValueError('모델 버전·동작·샘플링률·필터 설정이 일치하지 않습니다.')
    for name, shape in [('center', (8,)), ('scale', (8,)), ('weights', (3, 8)), ('intercept', (3,))]:
        a = np.asarray(model[name], float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError('잘못된 모델: '+name)
    if np.any(np.array(model['scale']) <= 0):
        raise ValueError('잘못된 모델 scale')
    return model


def evaluate(records, model, test_rounds):
    selected = [r for r in records if r['phase']=='hold' and r['round'] in test_rounds]
    if not selected:
        raise ValueError('평가 데이터가 없습니다.')
    pred, _ = predict(model, [r['feature'] for r in selected])
    truth = np.array([r['label'] for r in selected])
    confusion = np.zeros((3, 3), int)
    np.add.at(confusion, (truth, pred), 1)
    trials = []
    for event in sorted(set(r['event_id'] for r in selected)):
        mask = np.array([r['event_id']==event for r in selected])
        vote = int(np.argmax(np.bincount(pred[mask], minlength=3)))
        row = selected[int(np.flatnonzero(mask)[0])]
        trials.append(dict(round=row['round'], label=row['label'], prediction=vote,
                           windows=int(mask.sum())))
    return dict(window_confusion=confusion.tolist(), window_accuracy=float(np.mean(pred==truth)),
                trial_accuracy=float(np.mean([r['label']==r['prediction'] for r in trials])),
                trials=trials, missing_test_classes=[LABELS[k] for k in range(3) if not np.any(truth==k)],
                rest_false_activation=float(np.mean(pred[truth==0]!=0)) if np.any(truth==0) else None)


class GestureRecorder(Recorder):
    PATTERNS = np.array([[1]*8, [5, 4, 3, 1, 1, 1, 2, 2], [1, 1, 1, 2, 4, 5, 3, 2]])

    def sim_gain(self, event):
        label = event['label'] if event['phase'] == 'hold' else 0
        if event['phase'] == 'live':label = int((time.monotonic()-self.origin)//3) % 3
        return self.PATTERNS[max(0, label)]


class GestureApp:
    def __init__(self,args):
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.widgets import Button
        self.args=args
        available={f.name for f in font_manager.fontManager.ttflist}
        for f in ['AppleGothic','Malgun Gothic','NanumGothic','Noto Sans CJK KR']:
            if f in available:
                plt.rcParams['font.family']=f
                break
        plt.rcParams['axes.unicode_minus']=False
        self.model=load_model(args.model,args.rate) if args.model else None
        self.directory=session_dir(args.output,'gesture')
        self.plan=make_plan(args.rounds)
        self.index=0
        self.phase='waiting_calibration'
        self.phase_start=time.monotonic()
        self.finished=False
        self.durations=dict(calibration=20.,settle=3.,hold=3.,rest=5.)
        self.meta=dict(simulated=args.simulate,rate=args.rate,port=args.port,baud=args.baud,filter=filter_settings(args.rate),
                       status='recording',rounds=args.rounds,train_rounds=list(range(1,args.rounds-1)),
                       test_rounds=list(range(args.rounds-1,args.rounds+1)),plan=self.plan,completed_events=[],
                       model_source=str(Path(args.model).resolve()) if args.model else None,
                       time_semantics='PC read-completion timestamps; uniform nominal EMG rate within feature windows',
                       durations=self.durations)
        save_json(self.directory/'metadata.json',self.meta)
        self.reader=GestureRecorder(self.directory,args)
        self.fig,(self.bar_ax,self.result_ax)=plt.subplots(1,2,figsize=(14,9))
        self.fig.subplots_adjust(top=.59,bottom=.22,left=.07,right=.96,wspace=.25)
        self.fig.suptitle('EMG 손동작 학습·평가'+(' [가상 데이터]' if args.simulate else ''),fontsize=18)
        self.instruction=self.fig.text(.06,.91,'밴드를 착용하고 팔을 받치세요. 손목을 고정하고 손의 힘을 빼세요.',fontsize=13)
        self.detail=self.fig.text(.06,.865,'시작을 누르면 3초 준비 후 20초 휴식 기준값과 자이로 영점을 기록합니다.',fontsize=11)
        self.progress=self.fig.text(.06,.815,'준비 대기',fontsize=12,color='#166534')
        self.prediction=self.fig.text(.06,.755,'학습 전 · 동작 수집부터 진행하세요',fontsize=14)
        self.status=self.fig.text(.06,.685,'연결 중',fontsize=9)
        self.bars=self.bar_ax.bar(range(1,9),np.zeros(8),color='#2563eb')
        self.bar_ax.set(title='채널별 250ms RMS',xlabel='EMG 채널',ylabel='Filtered counts')
        self.result_ax.axis('off')
        self.result_text=self.result_ax.text(0,1,'수집 순서\n\n휴식 · 주먹 · 손가락 펴기\n매 라운드 순서를 섞어서 진행\n\n각 동작 3초 + 휴식 5초\n라운드 사이에는 준비 완료를 기다립니다.',va='top',fontsize=11)
        self.go=Button(self.fig.add_axes([.08,.11,.34,.06]),'Start / 준비 완료')
        self.stop=Button(self.fig.add_axes([.48,.11,.34,.06]),'Stop & Save / 중단·저장')
        self.go.on_clicked(self.ready)
        self.stop.on_clicked(lambda _:self.finish('aborted'))
        self.fig.text(.06,.04,'IMU는 함께 기록하지만 분류 입력에는 사용하지 않습니다 · 표시 점수는 정확도/신뢰확률이 아닙니다',fontsize=10)
        self.fig.canvas.mpl_connect('close_event',lambda _:self.finish('aborted'))
        self.timer=self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.tick)
        self.reader.start()
        self.timer.start()
        print('Gesture session:',self.directory,flush=True)

    def transition(self,phase):
        self.phase,self.phase_start=phase,time.monotonic()
        r,label=self.plan[min(self.index,len(self.plan)-1)]
        self.reader.set_phase(phase,r if phase not in ('calibration','settle_calibration') else 0,label if phase=='hold' else -1)
        if phase=='hold':
            self.instruction.set_text(KOREAN[label])
            self.detail.set_text('측정 중: 손목과 팔 위치를 유지하세요. 동작이 끝나면 휴식 안내가 나옵니다.')
        elif phase=='settle':
            self.instruction.set_text('다음 동작 → '+KOREAN[label])
            self.detail.set_text('준비 시간입니다. 자세를 맞추고 측정 시작을 기다리세요.')
        elif phase=='rest':
            self.instruction.set_text('휴식: 손의 힘을 빼고 편하게 쉬세요.')
            self.detail.set_text('휴식 시간이 끝나면 다음 동작 또는 다음 라운드 안내가 나옵니다.')
        elif phase=='waiting_round':
            self.instruction.set_text(f'{r}번째 라운드 준비 · 팔을 같은 위치에 받치세요.')
            warning=' · 정지 중 흔들림이 커서 자이로 보정은 미적용' if not self.meta.get('calibration',{}).get('applied',False) else ''
            self.detail.set_text('준비 완료를 누르면 3개 동작을 안내에 따라 진행합니다.'+warning)
            self.go.set_active(True)
        elif phase=='calibration':
            self.instruction.set_text('기준값 측정: 손과 손목에 힘을 빼고 움직이지 마세요.')
        elif phase=='live':
            self.instruction.set_text('수집·평가 완료. 이제 자유롭게 휴식 / 주먹 / 펴기를 해보세요.')
            self.detail.set_text('실시간 예측 중 · 완료하려면 Stop & Save를 누르세요. 원본 데이터는 계속 저장됩니다.')

    def ready(self,_=None):
        if self.finished or self.phase not in ('waiting_calibration','waiting_round'):return
        s=self.reader.snapshot()
        if s['last_emg'] is None or time.monotonic()-s['last_emg']>1 or s['latest'] is None:
            self.progress.set_text('EMG 수신과 필터 준비를 기다리세요. 다른 센서 프로그램은 종료하세요.')
            return
        if s['last_imu'] is None or time.monotonic()-s['last_imu']>1:
            self.progress.set_text('IMU 수신이 없습니다. 연결을 확인하세요.')
            return
        self.go.set_active(False)
        self.transition('settle_calibration' if self.phase=='waiting_calibration' else 'settle')

    def calibrate(self):
        with self.reader.lock:
            event=self.reader.events[-1]['event_id']
            values=np.array([v for t,e,v in self.reader.imu if e==event and t>self.reader.events[-1]['host_time_s']+.5])
            records=[r for r in self.reader.records if r['event_id']==event]
        if len(values)<10 or len(records)<2:raise ValueError('기준값 데이터가 부족합니다.')
        bias=values[:,:3].mean(0)
        gyro_std=values[:,:3].std(0)
        norm_std=np.linalg.norm(values[:,3:],axis=1).std()
        stable=bool(gyro_std.max()<.02 and norm_std<.1)
        self.meta['calibration']=dict(gyro_bias_candidate_rad_s=bias.tolist(),gyro_std=gyro_std.tolist(),
                                      accel_norm_std=float(norm_std),applied=stable,
                                      criterion='Heuristic: max gyro std < 0.02 rad/s and accel norm std < 0.1 m/s2',
                                      rest_rms_median=np.median([r['rms'] for r in records],axis=0).tolist())
        with self.reader.lock:
            self.reader.bias=bias if stable else np.zeros(3)
        save_json(self.directory/'metadata.json',self.meta)

    def build_results(self):
        with self.reader.lock:
            records=[r for r in self.reader.records if r['event_id'] in self.meta['completed_events']]
        train=[r for r in records if r['phase']=='hold' and r['round'] in self.meta['train_rounds']]
        if not self.args.model:
            for round_id in self.meta['train_rounds']:
                if set(r['label'] for r in train if r['round']==round_id)!={0,1,2}:
                    raise ValueError('학습 라운드에 유효 동작 구간이 누락됐습니다.')
            self.model=fit_model([r['feature'] for r in train],[r['label'] for r in train],self.args.rate)
        test_rounds=list(range(1,self.args.rounds+1)) if self.args.model else self.meta['test_rounds']
        metrics=evaluate(records,self.model,test_rounds)
        if metrics['missing_test_classes'] or len(metrics['trials'])!=3*len(test_rounds):
            raise ValueError('평가 라운드에 유효 동작 구간이 누락됐습니다.')
        self.model['simulated_training']=self.model.get('simulated_training',self.args.simulate)
        save_json(self.directory/'model.json',self.model)
        metrics.update(labels=LABELS,train_rounds=[] if self.args.model else self.meta['train_rounds'],test_rounds=test_rounds,
                       simulated=self.args.simulate,simulated_training=self.model['simulated_training'],
                       note='Frozen training-only model. Overlapping windows are correlated. Trial and new-session results matter.')
        metrics['median_harmonic_fraction']={LABELS[k]:np.median([r['harmonic_fraction'] for r in records if r['phase']=='hold' and r['label']==k],axis=0).tolist() for k in range(3)}
        save_json(self.directory/'analysis.json',metrics)
        matrix=np.array(metrics['window_confusion'])
        lines=['# 손동작 평가'+(' [가상 데이터]' if self.args.simulate else ''),'',
               f"완료 동작 단위 정확도: {metrics['trial_accuracy']:.1%}",f"윈도 단위 정확도: {metrics['window_accuracy']:.1%}",
               f"휴식 윈도의 힘주기 오인식률: {metrics['rest_false_activation']:.1%}",'',
               '혼동행렬: 행=실제, 열=예측. 순서는 휴식 / 주먹 / 펴기.',str(matrix),'',
               '학습에 사용하지 않은 마지막 두 라운드로 평가합니다. 외부 모델을 불러온 경우 모든 라운드가 평가용입니다.',
               '겹치는 윈도는 독립 표본이 아닙니다. 데이터가 적으므로 같은 착용 상태의 높은 점수가 재착용 성능을 보장하지 않습니다.',
               '모델은 보류 평가 후 전체 데이터로 재학습하지 않습니다. model.json을 다음 세션에서 그대로 불러올 수 있습니다.',
               '60Hz 배수 주변 전력 비율은 잔여 간섭 참고 지표이며 정확한 SNR이나 합격 기준이 아닙니다.',
               'IMU는 분류 입력에서 제외했습니다. 손목 움직임과 동작의 동반 여부는 imu.csv를 함께 확인하세요.',
               'calibration의 applied가 false이면 정지 조건을 만족하지 않아 자이로 보정을 적용하지 않았습니다.',
               '가속도 영점/배율은 보정하지 않습니다. EMG 단위는 counts입니다.']
        (self.directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        self.result_text.set_text(f"평가 결과 (학습에 쓰지 않은 데이터)\n\n동작 단위: {metrics['trial_accuracy']:.1%}\n윈도 단위: {metrics['window_accuracy']:.1%}\n휴식 오인식: {metrics['rest_false_activation']:.1%}\n\n행=실제 / 열=예측\n휴식 · 주먹 · 펴기\n{matrix}\n\nreport.md / analysis.json / model.json 저장")
        self.meta['evaluation_complete']=True
        save_json(self.directory/'metadata.json',self.meta)

    def tick(self):
        if self.finished:return
        s=self.reader.snapshot()
        if s['error']:
            self.finish('error');return
        now=time.monotonic()
        if not self.phase.startswith('waiting') and any(s[k] is None or now-s[k]>2 for k in ['last_emg','last_imu']):
            self.finish('signal_lost');return
        duration=self.durations.get('settle' if self.phase.startswith('settle') else self.phase)
        if duration is not None and now-self.phase_start>=duration:
            try:
                if self.phase=='settle_calibration':self.transition('calibration')
                elif self.phase=='calibration':
                    self.calibrate();self.transition('waiting_round')
                elif self.phase=='settle':self.transition('hold')
                elif self.phase=='hold':
                    self.meta['completed_events'].append(self.reader.events[-1]['event_id'])
                    save_json(self.directory/'metadata.json',self.meta)
                    self.transition('rest')
                elif self.phase=='rest':
                    self.index+=1
                    if self.index==len(self.plan):
                        self.build_results();self.transition('live')
                    else:self.transition('waiting_round' if self.index%3==0 else 'settle')
            except Exception as error:
                self.meta['analysis_error']=str(error)
                self.finish('analysis_error');return
        duration=self.durations.get('settle' if self.phase.startswith('settle') else self.phase)
        if duration is not None:
            cue={'settle_calibration':'기준값 측정 준비','calibration':'기준값 측정 중','settle':'다음 동작 준비','hold':'동작 측정 중','rest':'휴식'}[self.phase]
            self.progress.set_text(f'라운드 {self.plan[min(self.index,len(self.plan)-1)][0]}/{self.args.rounds} · {cue} · 남은 시간 {max(0,duration-(now-self.phase_start)):.1f}초')
        else:self.progress.set_text('실시간 예측 중' if self.phase=='live' else '준비 완료 버튼을 누르세요')
        latest=s['latest']
        fresh=latest is not None and now-self.reader.origin-latest['end_time_s']<.5
        if fresh:
            for b,value in zip(self.bars,latest['rms']):b.set_height(value)
            self.bar_ax.set_ylim(0,max(1,max(latest['rms'])*1.2))
        if self.model and fresh:
            pred,p=predict(self.model,[latest['feature']])
            self.prediction.set_text(f'예측: {KOREAN[int(pred[0])].split(":")[0]} · 모델 점수 {p[0,int(pred[0])]:.0%} (보정된 확률 아님)')
        elif self.model:self.prediction.set_text('예측 대기: 신호 수신/필터 준비 중')
        corrected=''
        if s['imu']:
            g=np.array(s['imu'][2][:3])-self.reader.bias
            corrected=' · Gyro '+', '.join(f'{v:+.3f}' for v in g)+' rad/s'
        applied=self.meta.get('calibration',{}).get('applied')
        self.status.set_text(f"EMG {s['aa']:,} · IMU {s['bb']:,} · 불연속 {s['breaks']} · 자이로 보정 {applied}{corrected}")
        self.fig.canvas.draw_idle()

    def finish(self,status):
        if self.finished:return
        self.finished=True
        self.timer.stop()
        self.reader.set_phase('end')
        self.reader.stop_event.set()
        self.reader.join()
        state=self.reader.snapshot()
        if status=='aborted' and self.meta.get('evaluation_complete'):status='completed'
        self.meta.update(status='error' if state['error'] else status,error=state['error'],
                         elapsed_s=time.monotonic()-self.reader.origin,aa=state['aa'],bb=state['bb'],
                         counter_breaks=state['breaks'],discarded_bytes=self.reader.parser.discarded)
        save_json(self.directory/'metadata.json',self.meta)
        self.go.set_active(False)
        self.instruction.set_text('저장 완료 · '+self.meta['status'])
        self.detail.set_text(str(self.directory))
        self.progress.set_text(self.meta.get('analysis_error') or state['error'] or ('report.md에서 평가 결과를 확인하세요.' if self.meta.get('evaluation_complete') else '부분 기록 저장: 수집을 완료하지 않아 모델 평가를 생성하지 않았습니다.'))
        self.fig.canvas.draw_idle()
        print('Saved:',self.directory,flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='emg-probe gesture',description=__doc__)
    parser.add_argument('--port')
    parser.add_argument('--baud',type=int,default=BAUD)
    parser.add_argument('--rate',type=int,default=2000)
    parser.add_argument('--rounds',type=int,default=8)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--output',default=DEFAULT_OUTPUT)
    parser.add_argument('--simulate',action='store_true')
    args=parser.parse_args(argv)
    if args.rounds<4:parser.error('--rounds는 4 이상이어야 합니다.')
    if args.rate<=900 or args.baud<=0:parser.error('rate > 900, baud > 0 필요')
    if not args.port and not args.simulate:
        try:args.port=find_port()
        except ValueError as e:parser.error(str(e))
    import matplotlib.pyplot as plt
    app=GestureApp(args)
    try:plt.show()
    finally:app.finish('aborted')


if __name__=='__main__':main()
