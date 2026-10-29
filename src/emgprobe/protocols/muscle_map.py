"""Quick channel -> muscle map for one or two EMG bands (wrist/forearm band, upper-arm band).

Short isometric tasks (rest, wrist flexion/extension, radial/ulnar deviation, grip, elbow flexion palm-up/palm-down,
elbow extension), 3 repetitions in random order, both bands recorded together. Each channel is labelled with the task
that raises its RMS most (-> approximate muscle) and keeps a rest baseline and a task reference, used later to express
EMG as a fraction of the mapping response (emgprobe.analysis.normalize). Not MVC.
The placement check compares wrist-task and elbow-task responses of the two bands (catches swapped ports).
Valid only for the donning it was recorded in: re-run after taking a band off or shifting it.

usage: emg-probe map --band wrist=/dev/cu.A --band upperarm=/dev/cu.B
       emg-probe map --simulate --band wrist --band upperarm
recompute: emg-probe map --analyse recordings/muscle_map_...
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emgprobe import DEFAULT_OUTPUT
from emgprobe.analysis import phase_windows
from emgprobe.bands import Bands, ROLES, parse_bands
from emgprobe.device import BAUD
from emgprobe.files import save_json, session_dir
from emgprobe.filters import filter_settings
from emgprobe.recorder import WindowRecorder
from emgprobe.terminal import timed, wait_for_data

POSTURE = ('앉아서 팔꿈치 약 90°, 아래팔을 책상 위에 올리고 엄지가 위를 향하게(중립) 두세요. '
           '저항은 반대손이나 책상으로 주고, 관절이 움직이지 않게 힘만 내세요(등척성).')
# label = index; label 0 is the rest task.
TASKS = [
    dict(id='rest', name='휴식', cue='손과 팔의 힘을 모두 빼세요.'),
    dict(id='wrist_flex', name='손목 굽힘', cue='반대손으로 손바닥을 막고, 손바닥 쪽으로 손목을 굽히려는 힘.'),
    dict(id='wrist_ext', name='손목 폄', cue='반대손으로 손등을 막고, 손등 쪽으로 손목을 펴려는 힘.'),
    dict(id='radial_dev', name='엄지 쪽 꺾기', cue='엄지가 위인 자세에서 반대손으로 엄지 쪽을 막고, 손을 위(엄지 쪽)로 들려는 힘.'),
    dict(id='ulnar_dev', name='새끼 쪽 꺾기', cue='반대손으로 새끼손가락 쪽을 막고, 손을 아래(새끼 쪽)로 내리려는 힘.'),
    dict(id='grip', name='주먹 쥐기', cue='손목은 중립 그대로, 주먹(또는 수건)을 꽉 쥐기.'),
    dict(id='elbow_flex_sup', name='팔꿈치 굽힘 (손바닥 위)', cue='손바닥을 위로 하고 책상 밑면(또는 반대손)을 들어 올리듯 당기기.'),
    dict(id='elbow_flex_pro', name='팔꿈치 굽힘 (손바닥 아래)', cue='손바닥을 아래로 하고 책상 밑면(또는 반대손)을 들어 올리듯 당기기.'),
    dict(id='elbow_ext', name='팔꿈치 폄', cue='주먹 새끼손가락 쪽을 책상 위에 대고 아래로 누르기 (팔꿈치를 펴려는 힘).'),
]
TASK_IDS = [t['id'] for t in TASKS]
EFFORT = '세기: 최대의 절반 정도, 아프지 않게, 매번 비슷하게.'
# Task -> (group key, approximate muscle) per band. Approximate: surface EMG under a band also picks up neighbours.
MUSCLES = {
    'wrist': dict(wrist_flex=('flexors', '손목 굴곡근 (FCR·PL·FCU)'), wrist_ext=('extensors', '손목 신전근 (ECRL·ECRB·ECU)'),
                  radial_dev=('radial', '엄지 쪽 (ECRL·FCR·완요골근)'), ulnar_dev=('ulnar', '새끼 쪽 (FCU·ECU)'),
                  grip=('finger_flexors', '손가락 굴곡근 (FDS)'), elbow_flex_sup=('brachioradialis', '완요골근 (BR)'),
                  elbow_flex_pro=('brachioradialis', '완요골근 (BR)'), elbow_ext=('elbow_ext', '팔꿈치 폄 반응 (주근 등)')),
    'upperarm': dict(wrist_flex=('wrist_tasks', '불명확 (손목 동작에 반응)'), wrist_ext=('wrist_tasks', '불명확 (손목 동작에 반응)'),
                     radial_dev=('wrist_tasks', '불명확 (손목 동작에 반응)'), ulnar_dev=('wrist_tasks', '불명확 (손목 동작에 반응)'),
                     grip=('wrist_tasks', '불명확 (손목 동작에 반응)'), elbow_flex_sup=('biceps', '이두근'),
                     elbow_flex_pro=('brachialis', '상완근 (이두 아래·옆)'), elbow_ext=('triceps', '삼두근')),
}
WRIST_TASKS, ELBOW_TASKS = TASK_IDS[1:6], TASK_IDS[6:]
MAP_S = dict(baseline=8., settle=1., hold=4., rest=3.)
TRIM_S = 1.   # windows ending within 1 s of a phase start are dropped (reaction + 100 ms window)


class MapRecorder(WindowRecorder):
    # All phases keep their 100 ms windows; analysis trims phase starts itself.
    FEATURE_PHASES = ('baseline', 'hold', 'pre', 'task', 'post')
    FEATURE_SKIP_S = 0.

    def sim_gain(self, e):
        if e['phase'] not in ('hold', 'task') or e['label'] <= 0:return np.ones(8)
        # Every task excites its own channel, shifted by one channel on the upper-arm band.
        centre = (3*e['label']+(self.directory.name == 'upperarm')) % 8
        return 1+4*np.exp(-.5*((np.arange(8)-centre+4) % 8-4)**2)


def mapping_plan(reps, seed=2026):
    rng = np.random.default_rng(seed)
    return [dict(rep=r, label=int(k)) for r in range(1, reps+1) for k in rng.permutation(len(TASKS))]


def run_mapping(a, directory):
    """Terminal mapping session in `directory` (one sub-folder per band). Returns the analysis."""
    directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
    plan = mapping_plan(a.map_reps)
    meta = dict(kind='muscle_mapping', status='recording', simulated=a.simulate, rate=a.rate, baud=a.baud, filter=filter_settings(a.rate),
                reps=a.map_reps, durations_s=dict(MAP_S), tasks=TASKS, plan=plan, completed_events=[], note=a.note,
                features='100ms RMS, 100ms stride; analysis drops windows ending < 1 s after a phase start')
    group = Bands(MapRecorder, directory, SimpleNamespace(port=None, baud=a.baud, rate=a.rate, simulate=a.simulate), a.bands)
    group.save_meta(meta);group.start()
    try:
        wait_for_data(group)
        print(f'\n=== 근육 매핑: {len(TASKS)}개 동작 x {a.map_reps}회, 약 {len(plan)*sum(MAP_S[k] for k in ("settle", "hold", "rest"))/60+.5:.0f}분 ===')
        print(POSTURE+'\n'+EFFORT)
        input(f'기준선: 힘을 빼고 {MAP_S["baseline"]:.0f}초 가만히. ENTER로 시작 ')
        timed(group, 'baseline', 0, 0, MAP_S['baseline'], '기준선')
        for i, step in enumerate(plan):
            task = TASKS[step['label']]
            group.set_phase('waiting', step['rep'], step['label'])
            input(f"\n[{i+1}/{len(plan)}] {task['name']}: {task['cue']}\n  ENTER로 시작 (Ctrl+C: 중단) ")
            timed(group, 'settle', step['rep'], step['label'], MAP_S['settle'], '  힘 주기 시작')
            timed(group, 'hold', step['rep'], step['label'], MAP_S['hold'], f"  {task['name']} 유지")
            meta['completed_events'].append(group.primary.events[-1]['event_id']);group.save_meta(meta)
            timed(group, 'rest', step['rep'], 0, MAP_S['rest'], '  힘 빼고 쉬기')
        meta['status'] = 'completed'
    except (KeyboardInterrupt, EOFError):
        meta['status'] = 'aborted'
    except RuntimeError as e:
        meta.update(status='error', error=str(e))
    finally:
        group.set_phase('end');group.stop();s = group.snapshot()
        meta.update(status='error' if s['error'] else meta['status'], error=s['error'] or meta.get('error'),
                    counter_breaks=s['breaks'], discarded_bytes=group.discarded())
        group.save_meta(meta)
    result = analyse_mapping(directory)
    print_mapping(result)
    print('saved', directory)
    return result


def analyse_band(band_dir, role, completed):
    holds = phase_windows(band_dir, 'hold', TRIM_S)
    rest = [r['rms'] for w in phase_windows(band_dir, 'baseline', TRIM_S).values() for r in w]
    reps = {}   # task id -> [(rep, median rms)]
    for ev in completed:
        w = holds.get(ev, [])
        if len(w) < 5:continue
        task = TASK_IDS[w[0]['label']]
        reps.setdefault(task, []).append((w[0]['round'], np.median([r['rms'] for r in w], 0)))
        if task == 'rest':rest += [r['rms'] for r in w]
    active = [t for t in TASK_IDS[1:] if t in reps]
    if len(rest) < 10 or not active:return dict(error='Not enough baseline or task data', tasks_done=sorted(reps))
    rest = np.array(rest);b = np.median(rest, 0);noise = 1.4826*np.median(abs(rest-b), 0)
    resp = np.array([np.median([m for _, m in reps[t]], 0)-b for t in active])   # tasks x channels
    ref = resp.max(0);best = resp.argmax(0)
    valid = ref > np.maximum(3*noise, .25*b)
    profile = np.clip(resp/np.where(ref > 0, ref, np.nan), 0, None)
    # Repetition consistency: in how many repetitions (present for every task) the same task wins.
    by_rep = {}
    for t in active:
        for rep, m in reps[t]:by_rep.setdefault(rep, {})[t] = m-b
    full = [v for v in by_rep.values() if len(v) == len(active)]
    wins = [np.array([v[t] for t in active]).argmax(0) for v in full]
    pos = lambda t: resp[active.index(t)] if t in active else np.zeros(8)
    names = MUSCLES.get(role, MUSCLES['wrist'])
    channels = []
    for i in range(8):
        t = active[best[i]]
        if role == 'upperarm':side = '앞(이두쪽)' if pos('elbow_flex_sup')[i] > pos('elbow_ext')[i] else '뒤(삼두쪽)'
        else:side = ('손바닥쪽' if pos('wrist_flex')[i] > pos('wrist_ext')[i] else '손등쪽')+'/'+('엄지쪽' if pos('radial_dev')[i] > pos('ulnar_dev')[i] else '새끼쪽')
        channels.append(dict(channel=i+1, valid=bool(valid[i]), best_task=t, group=names[t][0] if valid[i] else 'weak', muscle=names[t][1] if valid[i] else '신호 약함',
                             share=float(resp[best[i], i]/max(np.clip(resp[:, i], 0, None).sum(), 1e-9)),
                             consistency=f'{sum(int(w[i] == best[i]) for w in wins)}/{len(wins)}', side=side,
                             profile={t: float(profile[k, i]) for k, t in enumerate(active)}))
    score = lambda tasks: float(np.mean([np.nanmax([profile[active.index(t), i] for t in tasks if t in active] or [0]) for i in np.flatnonzero(valid)])) if valid.any() else None
    groups = {}
    for c in channels:
        if c['valid']:groups.setdefault(c['group'], []).append(c['channel'])
    return dict(baseline=b.tolist(), reference=(b+ref).tolist(), noise=noise.tolist(), valid=valid.tolist(), tasks=active,
                channels=channels, groups=groups, wrist_task_score=score(WRIST_TASKS), elbow_task_score=score(ELBOW_TASKS),
                repetitions={t: len(v) for t, v in reps.items()})


def judge_placement(bands):
    """Wrist band should respond more to wrist tasks, upper-arm band more to elbow tasks; both reversed -> ports swapped."""
    w, u = bands.get('wrist', {}), bands.get('upperarm', {})
    if any(x.get('wrist_task_score') is None for x in (w, u)):return 'unknown'
    wrist_ok, upper_ok = w['wrist_task_score'] > w['elbow_task_score'], u['elbow_task_score'] > u['wrist_task_score']
    return 'ok' if wrist_ok and upper_ok else 'swapped?' if not wrist_ok and not upper_ok else 'unclear'


def analyse_mapping(directory):
    p = Path(directory);meta = json.loads((p/'metadata.json').read_text())
    bands = {role: analyse_band(p/role, role, meta['completed_events']) for role in meta['bands']}
    result = dict(kind='muscle_map', source=str(p.resolve()), status=meta['status'], simulated=meta['simulated'], filter=meta['filter'],
                  rms_window_ms=100, bands=bands, placement=judge_placement(bands),
                  note='reference = baseline + largest median task response (moderate isometric effort, NOT MVC). '
                       'group/muscle = task that raises the channel most; approximate (crosstalk from neighbouring muscles).')
    save_json(p/'muscle_map.json', result)
    (p/'muscle_map.md').write_text(report(result))
    plot_mapping(result, p/'muscle_map.png')
    return result


def report(result):
    lines = ['# Muscle map', '', f"source: `{result['source']}` | status {result['status']} | placement check: **{result['placement']}**", '']
    for role, b in result['bands'].items():
        lines += [f'## {role}', '']
        if 'error' in b:lines += [b['error'], ''];continue
        lines += ['| ch | 근육 (추정) | 위치 | best task | share | 반복 일치 |', '|---|---|---|---|---|---|']
        lines += [f"| {c['channel']} | {c['muscle']} | {c['side']} | {c['best_task']} | {c['share']:.2f} | {c['consistency']} |" for c in b['channels']]
        lines += ['', 'groups: '+', '.join(f'{g}={chs}' for g, chs in b['groups'].items()), '']
    return '\n'.join(lines)


def print_mapping(result):
    print(f"\n=== 매핑 결과 (placement check: {result['placement']}) ===")
    if result['placement'] == 'swapped?':print('  !! 손목 밴드가 팔꿈치 동작에, 위팔 밴드가 손목 동작에 더 반응합니다. 두 포트가 바뀌었는지 확인하세요.')
    for role, b in result['bands'].items():
        if 'error' in b:print(f'{role}: {b["error"]}');continue
        print(f'{role}:')
        for c in b['channels']:print(f"  ch{c['channel']}  {c['muscle']:<24} {c['side']:<10} share {c['share']:.2f}  반복 {c['consistency']}")


def plot_mapping(result, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    bands = {r: b for r, b in result['bands'].items() if 'error' not in b}
    if not bands:return
    fig, axes = plt.subplots(1, len(bands), figsize=(6*len(bands), 4.2), squeeze=False, layout='constrained')
    for ax, (role, b) in zip(axes[0], bands.items()):
        m = np.array([[c['profile'].get(t, np.nan) for c in b['channels']] for t in b['tasks']])
        im = ax.imshow(m, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_xticks(range(8), [f"{c['channel']}{'' if c['valid'] else '*'}" for c in b['channels']])
        ax.set_yticks(range(len(b['tasks'])), b['tasks'])
        ax.set_xlabel('channel (* = weak)');ax.set_title(f'{role}: response / channel max')
    fig.colorbar(im, ax=axes[0].tolist(), shrink=.8)
    fig.savefig(path, dpi=110, bbox_inches='tight');plt.close(fig)


def load_map(path, rate=None, simulate=None):
    """Bands of a muscle map ({role: band}, bands without enough data left out) from a mapping folder or muscle_map.json.
    With `rate`, the map's filter must match; with simulate=False, a simulated map is refused."""
    p = Path(path);p = p/'muscle_map.json' if p.is_dir() else p
    m = json.loads(p.read_text())
    if simulate is False and m.get('simulated') is not False:raise ValueError(f'{p} is a simulated map; map this donning with real bands')
    if rate is not None and m.get('filter') != filter_settings(rate):raise ValueError(f'{p} was made with other filter settings; re-run the mapping')
    return {r: b for r, b in m['bands'].items() if 'error' not in b}


def main(argv=None):
    p = argparse.ArgumentParser(prog='emg-probe map', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--band', action='append', default=[], metavar='ROLE=PORT', help=f'ROLE in {list(ROLES)}; port optional with --simulate')
    p.add_argument('--simulate', action='store_true')
    p.add_argument('--baud', type=int, default=BAUD)
    p.add_argument('--rate', type=int, default=2000)
    p.add_argument('--map-reps', type=int, default=3)
    p.add_argument('--note', default='')
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--analyse', type=Path, help='Recompute muscle_map.json/.md/.png of a saved mapping folder')
    a = p.parse_args(argv)
    if a.analyse:print_mapping(analyse_mapping(a.analyse));return
    try:a.bands = parse_bands(a.band, a.simulate)
    except ValueError as e:p.error(str(e))
    if not a.bands:p.error('Give --band wrist=PORT and/or --band upperarm=PORT')
    if a.map_reps < 2:p.error('--map-reps must be >= 2')
    run_mapping(a, session_dir(a.output, 'muscle_map'))


if __name__ == '__main__':
    main()
