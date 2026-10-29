"""Subjective effort levels (rest / low / medium / higher, slow ramps up and down) with a GUI; one band.

No force or stiffness ground truth is measured or inferred: labels are instructed effort, not N or %MVC.
usage: emg-probe effort [--port /dev/cu.A] [--simulate] [--rounds 5]
recompute: emg-probe effort --analyse recordings/effort_...
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

from emgprobe import DEFAULT_OUTPUT
from emgprobe.bands import Bands, parse_bands
from emgprobe.device import BAUD, find_port
from emgprobe.files import save_json, session_dir
from emgprobe.filters import filter_settings
from emgprobe.recorder import Recorder

NAMES = ['휴식', '약하게', '중간', '조금 강하게', '서서히 증가', '서서히 감소']


def protocol(rounds):
    rng = np.random.default_rng(2026)
    plan = [dict(round=r, label=int(k), kind='steady', seconds=4.)
            for r in range(1, rounds+1) for k in rng.permutation(4)]
    plan += [dict(round=rounds+1,label=k,kind='ramp',seconds=8.) for k in (4,5)]
    return plan


class EffortRecorder(Recorder):
    def sim_gain(self, event):
        level = 0
        if event['phase'] == 'hold':
            label = event['label']
            level = label if label < 4 else np.clip((time.monotonic()-self.origin-event['host_time_s'])/8, 0, 1)*3
            if label == 5:level = 3-level
        return 1+level*np.array([1, .8, 1.2, .9, .7, 1.1, .6, 1])


def analyse(directory):
    p=Path(directory)
    meta=json.loads((p/'metadata.json').read_text())
    records=[json.loads(l) for l in (p/'features.jsonl').read_text().splitlines()] if (p/'features.jsonl').exists() else []
    imu={}
    if (p/'imu.csv').exists():
        with (p/'imu.csv').open() as f:
            for r in csv.DictReader(f):
                if r['phase']=='hold':
                    imu.setdefault(int(r['event_id']),[]).append([float(r[k]) for k in ['host_time_s','gx_rad_s','gy_rad_s','gz_rad_s']])
    trials=[]
    for event in sorted({r['event_id'] for r in records if r['phase']=='hold'}):
        rows=[r for r in records if r['event_id']==event]
        label=rows[0]['label']
        values=np.array([r['rms'] for r in rows])
        motion=np.array(imu.get(event,[])).reshape(-1,4)
        motion=motion[motion[:,0]>=rows[0]['host_time_s']+.5]
        bias=meta.get('calibration',{}).get('applied_bias_rad_s',[0,0,0])
        gyro=float(np.sqrt(np.mean(np.sum((motion[:,1:]-bias)**2,axis=1)))) if len(motion) else None
        trial=dict(event_id=event,round=rows[0]['round'],label=label,completed=event in meta['completed_events'],
                   windows=len(rows),median_rms= np.median(values,axis=0).tolist(),gyro_rms_rad_s=gyro)
        if label>=4 and len(rows)>=3:
            corr=spearmanr([r['end_time_s'] for r in rows],np.median(values,axis=1)).statistic
            trial['time_rms_spearman']=float(corr) if np.isfinite(corr) else None
        trials.append(trial)
    levels=[]
    for label in range(4):
        v=np.array([r['median_rms'] for r in trials if r['label']==label and r['completed']]).reshape(-1,8)
        levels.append(dict(label=label,repetitions=len(v),median_rms=np.median(v,axis=0).tolist() if len(v) else None,
                           between_trial_cv=(v.std(0,ddof=1)/np.maximum(v.mean(0),1e-9)).tolist() if len(v)>1 else None))
    ordered=[]
    for rd in range(1,meta['rounds']+1):
        group={r['label']:r for r in trials if r['round']==rd and r['label']<4 and r['completed']}
        if len(group)==4:
            values=np.array([group[k]['median_rms'] for k in range(4)])
            ordered.append(dict(round=rd,strictly_increasing_channels=(np.diff(values,axis=0)>0).all(0).tolist()))
    result=dict(status=meta['status'],simulated=meta['simulated'],label_semantics='Subjective instructed effort, NOT measured force or stiffness',
                measured_force_N=None,measured_stiffness=None,trials=trials,levels=levels,ordering_by_round=ordered,
                note='RMS includes residual interference; monotonicity does not prove force accuracy. Overlapping windows are correlated.')
    save_json(p/'analysis.json',result)
    with (p/'trial_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['event_id','round','subjective_label','completed','windows','gyro_rms_rad_s',*[f'rms_ch{i+1}' for i in range(8)]])
        for r in trials:w.writerow([r['event_id'],r['round'],NAMES[r['label']],r['completed'],r['windows'],r['gyro_rms_rad_s'],*r['median_rms']])
    lines=['# 주관적 힘주기 결과','',
           '**실측 힘·stiffness가 없는 실험입니다. 약·중·강은 안내 라벨이며 N 또는 %MVC가 아닙니다.**','',
           f"상태: {meta['status']} / 가상 데이터: {meta['simulated']} / 완료 구간: {len(meta['completed_events'])}",
           f"착용·물체·자세 메모: {meta.get('note','')}",f"수신 오류: {meta.get('error')} / 카운터 불연속: {meta.get('counter_breaks',0)}",'',
           '| 수준 | 완료 반복 | 채널별 RMS 중앙값 (counts) |','|---|---:|---|']
    for r in levels:lines.append(f"| {NAMES[r['label']]} | {r['repetitions']} | "+(', '.join(f'{v:.1f}' for v in r['median_rms']) if r['median_rms'] else '자료 없음')+' |')
    lines+=['','## 해석','',
            '- analysis.json의 between_trial_cv는 완료된 반복별 RMS 중앙값의 변동계수입니다. 사용자 힘 재현성과 신호 변화가 함께 반영됩니다.',
            '- ordering_by_round는 휴식→약→중→강으로 RMS가 증가한 채널입니다. 자동 합격 판정이나 실제 힘 보정이 아닙니다.',
            '- 서서히 증가/감소 구간의 시간-RMS 상관은 안내에 따른 추세 확인용이며 힘 센서와의 상관이 아닙니다.',
            '- gyro_rms_rad_s는 손목 움직임 참고값입니다. IMU로 움직이지 않는 근육 긴장이나 전극 접촉 변화를 모두 판별할 수 없습니다.',
            '- 불완료 구간은 trials에 남기되 수준별 비교에서는 제외합니다. 필터·특징 설정은 metadata.json에 보존됩니다.',
            '- force/stiffness 정답은 null입니다 (힘 센서 없음).']
    (p/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(12,7),layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(2,4)
    for ch,ax in enumerate(axes.flat):
        for r in trials:
            if r['completed'] and r['label']<4:ax.scatter(r['label'],r['median_rms'][ch],alpha=.6,s=18)
        med=[r['median_rms'][ch] if r['median_rms'] else np.nan for r in levels]
        ax.plot(range(4),med,color='black',lw=1)
        ax.set(title=f'CH{ch+1}',xticks=range(4),xticklabels=['Rest','Low','Mid','Higher'],ylabel='RMS counts')
        ax.grid(alpha=.2)
    fig.suptitle(('SIMULATION | ' if meta['simulated'] else '')+'Subjective effort: repeated trial RMS (not force)')
    fig.savefig(p/'analysis.png',dpi=120)
    return result


class EffortApp:
    def __init__(self,args):
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.widgets import Button,TextBox
        available={f.name for f in font_manager.fontManager.ttflist}
        for font in ['AppleGothic','Malgun Gothic','NanumGothic','Noto Sans CJK KR']:
            if font in available:plt.rcParams['font.family']=font;break
        plt.rcParams['axes.unicode_minus']=False
        self.args=args;self.plan=protocol(args.rounds);self.index=0;self.phase='waiting_calibration';self.finished=False
        self.prepare_s=3.;self.rest_s=5.;self.calibration_s=20.;self.started=time.monotonic()
        self.directory=session_dir(args.output,'effort')
        self.meta=dict(status='recording',simulated=args.simulate,rate=args.rate,port=args.port,baud=args.baud,rounds=args.rounds,
                       plan=self.plan,labels=NAMES,label_semantics='Subjective instructed effort; not N or %MVC',
                       force_sensor=None,measured_force_N=None,measured_stiffness=None,note=args.note,
                       filter=filter_settings(args.rate),features='250ms RMS, 100ms stride; first 0.5s after hold cue excluded',
                       timestamp_semantics='PC monotonic read-completion, shared by samples in a batch; nominal EMG rate for windows',
                       completed_events=[],prepare_s=self.prepare_s,rest_s=self.rest_s,calibration_s=self.calibration_s)
        save_json(self.directory/'metadata.json',self.meta)
        self.bands=Bands(EffortRecorder,self.directory,args,getattr(args,'bands',None));self.reader=self.bands.primary
        self.baseline=None;self.history=[];self.last_feature=None
        self.fig,(self.ax,self.trace)=plt.subplots(1,2,figsize=(14,9))
        self.fig.subplots_adjust(top=.61,bottom=.27,left=.07,right=.96,wspace=.28)
        self.fig.suptitle('EMG 힘주기 수준 · 반복성 테스트'+(' [가상 데이터]' if args.simulate else ''),fontsize=18)
        self.cue=self.fig.text(.06,.91,'같은 물체를 잡고 팔을 받치세요. 손목과 손가락 자세를 유지하세요.',fontsize=12)
        self.detail=self.fig.text(.06,.865,'메모에 물체·착용 위치·자세를 적고 준비 완료를 누르세요. 처음에는 20초 휴식 측정입니다.',fontsize=10)
        self.progress=self.fig.text(.06,.81,'수신 준비 중',fontsize=12,color='#166534')
        self.status=self.fig.text(.06,.745,'',fontsize=10)
        self.fig.text(.06,.68,'목표는 주관적인 힘주기 수준입니다. 그래프를 목표값에 맞추지 마세요. 최대 힘은 필요 없습니다.',fontsize=10)
        n=8*len(self.bands.readers);self.bars=self.ax.bar(range(1,n+1),np.zeros(n));self.ax.set(title='EMG 250ms RMS',xlabel='채널',ylabel='Filtered counts')
        if self.bands.multi:self.ax.set_xticks(range(1,n+1),[f'{role[0].upper()}{i}' for role in self.bands.readers for i in range(1,9)],fontsize=7)
        self.line,=self.trace.plot([],[]);self.trace.set(title='근활성 변화 (실제 힘 아님)',xlabel='PC 시간 (s)',ylabel='휴식 RMS 대비 채널별 비율의 중앙값')
        self.trace.grid(alpha=.2)
        self.box=TextBox(self.fig.add_axes([.12,.18,.79,.045]),'메모 ',initial=args.note)
        self.box.on_submit(self.set_note)
        self.ready_button=Button(self.fig.add_axes([.08,.085,.34,.055]),'Ready / 준비 완료')
        self.stop_button=Button(self.fig.add_axes([.49,.085,.34,.055]),'Stop & Save / 중단·저장')
        self.ready_button.on_clicked(self.ready);self.stop_button.on_clicked(lambda _:self.finish('aborted'))
        self.fig.text(.06,.025,'단계마다 자세를 확인하고 준비 완료 · 완료 시 자동 저장·분석 · EMG/IMU 원본 보존 · 힘·강성 모델은 학습하지 않습니다',fontsize=9)
        self.fig.canvas.mpl_connect('close_event',lambda _:self.finish('aborted'))
        self.timer=self.fig.canvas.new_timer(interval=100);self.timer.add_callback(self.tick)
        self.bands.start();self.timer.start()
        print('Effort session:',self.directory,flush=True)

    def save_meta(self):
        self.bands.save_meta(self.meta)

    def set_note(self,text):
        self.meta['note']=text
        self.save_meta()

    def transition(self,phase):
        self.phase=phase;self.started=time.monotonic()
        step=self.plan[min(self.index,len(self.plan)-1)]
        self.bands.set_phase(phase,step['round'] if phase in ('hold','settle','rest','waiting') else 0,step['label'] if phase=='hold' else -1)
        if phase in ('waiting','settle','hold'):
            label=step['label'];self.cue.set_text(f"단계 {self.index+1}/{len(self.plan)} · {NAMES[label]} · 같은 물체와 자세 유지")
            hints=['손의 힘을 빼세요. 물체는 받침에 놓아 떨어지지 않게 하세요.','가벼운 힘으로 쥐고 유지하세요.','편안한 중간 정도 힘으로 쥐고 유지하세요.','중간보다 조금 강하게 쥐세요. 최대 힘은 쓰지 마세요.',
                   '가벼운 힘에서 조금 강한 힘까지 천천히 높이세요.','조금 강한 힘에서 가벼운 힘까지 천천히 낮추세요.']
            self.detail.set_text(hints[label]+(' 준비 완료를 누르면 3초 후 측정합니다.' if phase=='waiting' else ''))
        elif phase=='rest':self.cue.set_text('휴식: 손의 힘을 빼고 쉬세요.');self.detail.set_text('다음 단계는 준비 완료를 누를 때까지 기다립니다.')
        elif phase=='calibration':self.cue.set_text('20초 휴식 기준 측정: 손과 손목에 힘을 빼고 움직이지 마세요.')
        elif phase=='prepare_calibration':self.cue.set_text('휴식 기준 측정 준비: 물체와 팔을 받치고 손의 힘을 빼세요.')
        self.ready_button.set_active(phase in ('waiting','waiting_calibration'))

    def ready(self,_=None):
        if self.finished or self.phase not in ('waiting_calibration','waiting'):return
        s=self.bands.snapshot();now=time.monotonic()
        if s['latest'] is None or any(s[k] is None or now-s[k]>1 for k in ['last_emg','last_imu']):
            self.progress.set_text('EMG+IMU 수신과 필터 준비를 기다리세요. 다른 센서 프로그램은 종료하세요.');return
        self.set_note(self.box.text)
        self.transition('prepare_calibration' if self.phase=='waiting_calibration' else 'settle')

    def calibrate(self):
        calibration={}
        for role,reader in self.bands.readers.items():
            with reader.lock:
                event=reader.events[-1]
                rows=[r for r in reader.records if r['event_id']==event['event_id']]
                v=np.array([v for t,e,v in reader.imu if e==event['event_id'] and t>event['host_time_s']+.5])
            if len(rows)<2 or len(v)<10:raise ValueError('휴식 기준 데이터가 부족합니다.'+(f' ({role})' if role else ''))
            baseline=np.maximum(np.median([r['rms'] for r in rows],axis=0),1e-6)
            bias=v[:,:3].mean(0);std=v[:,:3].std(0);norm_std=float(np.linalg.norm(v[:,3:],axis=1).std())
            applied=bool(std.max()<.02 and norm_std<.1)
            with reader.lock:reader.bias=bias if applied else np.zeros(3)
            calibration[role]=dict(rest_rms=baseline.tolist(),gyro_bias_candidate_rad_s=bias.tolist(),gyro_std=std.tolist(),
                                   accel_norm_std=norm_std,applied=applied,applied_bias_rad_s=reader.bias.tolist(),
                                   criterion='Heuristic max gyro std <0.02rad/s and accel norm std <0.1m/s2; not force calibration')
        self.baseline=np.concatenate([np.array(c['rest_rms']) for c in calibration.values()])
        self.meta['calibration']=calibration if self.bands.multi else calibration[None]
        self.save_meta()

    def tick(self):
        if self.finished:return
        s=self.bands.snapshot();now=time.monotonic()
        if s['error']:self.finish('error');return
        if not self.phase.startswith('waiting') and any(s[k] is None or now-s[k]>2 for k in ['last_emg','last_imu']):self.finish('signal_lost');return
        def duration():
            return {'prepare_calibration':self.prepare_s,'settle':self.prepare_s,'calibration':self.calibration_s,
                    'hold':self.plan[min(self.index,len(self.plan)-1)]['seconds'],'rest':self.rest_s}.get(self.phase)
        d=duration()
        if d is not None and now-self.started>=d:
            try:
                if self.phase=='prepare_calibration':self.transition('calibration')
                elif self.phase=='calibration':self.calibrate();self.transition('waiting')
                elif self.phase=='settle':self.transition('hold')
                elif self.phase=='hold':
                    self.meta['completed_events'].append(self.reader.events[-1]['event_id'])
                    self.save_meta();self.transition('rest')
                elif self.phase=='rest':
                    self.index+=1
                    if self.index==len(self.plan):self.finish('completed');return
                    self.transition('waiting')
            except Exception as e:self.meta['analysis_error']=str(e);self.finish('error');return
        d=duration()
        names={'prepare_calibration':'측정 준비','settle':'측정 준비','calibration':'휴식 기준 측정','hold':'측정 중','rest':'휴식'}
        self.progress.set_text(f'{names[self.phase]} · 남은 시간 {max(0,d-(now-self.started)):.1f}초' if d is not None else '준비 완료를 누르세요')
        latest=s['latest']
        if latest and now-self.reader.origin-latest['end_time_s']<.5:
            for b,v in zip(self.bars,latest['rms']):b.set_height(v)
            self.ax.set_ylim(0,max(1,max(latest['rms'])*1.2))
            if self.baseline is not None and latest['end_time_s']!=self.last_feature:
                self.last_feature=latest['end_time_s'];self.history.append((self.last_feature,float(np.median(np.array(latest['rms'])/self.baseline))))
                self.history=self.history[-300:];t,y=np.array(self.history).T;self.line.set_data(t,y)
                self.trace.set_xlim(max(0,t[-1]-30),max(30,t[-1]));self.trace.set_ylim(0,max(2,y.max()*1.1))
        cal=self.meta.get('calibration',{})
        cals=list(cal.values()) if self.bands.multi else [cal]
        correction='측정 전' if not cal else '적용' if all(c['applied'] for c in cals) else '미적용(일부/전체): 기준 측정 중 흔들림'
        self.status.set_text(f"EMG {s['aa']:,} · IMU {s['bb']:,} · 불연속 {s['breaks']} · 자이로 보정 {correction}")
        self.fig.canvas.draw_idle()

    def finish(self,status):
        if self.finished:return
        self.finished=True;self.timer.stop();self.bands.set_phase('end');self.bands.stop()
        s=self.bands.snapshot();error=s['error'] or self.meta.get('error')
        self.meta.update(status='error' if s['error'] else status,error=error,aa=s['aa'],bb=s['bb'],counter_breaks=s['breaks'],
                         discarded_bytes=self.bands.discarded(),elapsed_s=time.monotonic()-self.reader.origin,note=self.box.text)
        self.save_meta()
        self.cue.set_text('측정 완료 · 저장했습니다.' if status=='completed' else '중단 · 수집한 기록을 저장했습니다.')
        self.detail.set_text(str(self.directory))
        try:
            analyse(self.directory)
            self.progress.set_text('report.md: 반복성 결과 · analysis.png: 수준별 그래프 · analysis.json: 수치')
        except Exception as e:
            self.meta['analysis_error']=str(e);self.save_meta()
            self.progress.set_text('분석 오류: '+str(e))
        self.status.set_text(error or self.meta.get('analysis_error') or '약·중·강은 주관적 안내 라벨입니다. 실제 힘·stiffness를 측정한 결과가 아닙니다.')
        self.ready_button.set_active(False);self.box.set_active(False);self.fig.canvas.draw_idle()
        print('Saved:',self.directory,flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='emg-probe effort',description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port');parser.add_argument('--baud',type=int,default=BAUD);parser.add_argument('--rate',type=int,default=2000)
    parser.add_argument('--band',action='append',default=[],metavar='ROLE=PORT',help='Two bands at once instead of --port (port optional with --simulate)')
    parser.add_argument('--rounds',type=int,default=5);parser.add_argument('--note',default='')
    parser.add_argument('--output',default=DEFAULT_OUTPUT)
    parser.add_argument('--simulate',action='store_true');parser.add_argument('--analyse',type=Path)
    args=parser.parse_args(argv)
    if args.analyse:analyse(args.analyse);print('saved',args.analyse/'report.md');return
    if args.rounds<2 or args.rate<=900 or args.baud<=0:parser.error('rounds >=2, rate >900, baud >0 필요')
    try:
        args.bands=parse_bands(args.band,args.simulate) if args.band else None
        if args.bands and args.port:raise ValueError('Use either --port or --band')
        if not args.bands and not args.port and not args.simulate:args.port=find_port()
    except ValueError as e:parser.error(str(e))
    import matplotlib.pyplot as plt
    app=EffortApp(args)
    try:plt.show()
    finally:app.finish('aborted')


if __name__=='__main__':main()
