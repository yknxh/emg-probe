"""Stiffness-direction wiping demo: muscle mapping first, then N trials of each scenario, one or two EMG bands.

Each trial: ENTER -> pre (hold the tool still, 2 s) -> start sound -> task (default 8 s, pace sound every --pace-s) ->
stop sound -> post (1 s). Each trial is its own folder (trials/NNN_<scenario>_tKK/<band>/, single-band layout per band).
Between trials: ENTER next | r redo the last trial | n TEXT note on the last trial | q save and quit (--resume continues).
Scenario order: interleaved (default; within each setup, every round runs each scenario once in random order) or blocked.
See docs/stiffness_direction.md for band placement and the scenario rationale.

usage: emg-probe direction --band wrist=/dev/cu.A --band upperarm=/dev/cu.B
       emg-probe direction --simulate --band wrist --band upperarm --trials 2
       --list: scenarios | --scenarios all or H_light,H_stiff_z,... | --mapping <mapping folder> reuses a map (same donning only)
resume: --resume recordings/stiffness_direction_... --band ...   recompute: --analyse recordings/stiffness_direction_...
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emgprobe import DEFAULT_OUTPUT
from emgprobe.analysis import auc, normalize, phase_windows, task_imu
from emgprobe.bands import Bands, ROLES, parse_bands
from emgprobe.device import BAUD
from emgprobe.files import save_json, session_dir
from emgprobe.filters import filter_settings
from emgprobe.protocols.muscle_map import MapRecorder, load_map, print_mapping, run_mapping
from emgprobe.terminal import ask, beep, timed, wait_for_data

SETUPS = {
    'horizontal': '수평: 책상 위 보드(가능하면 아래에 저울). 테이프로 폭 3 cm, 길이 30 cm 레인 표시. 서서, 몸 정면에서 좌우로 닦기.',
    'vertical': '수직: 벽/세운 화이트보드, 레인 중심은 어깨 높이. 같은 레인 표시. 서서, 보드 정면에서 좌우로 닦기.',
}
COMMON = '같은 도구, 같은 쥐는 법. 레인을 따라 좌우로, 삑 소리마다 한 방향(stroke).'
SCENARIOS = [
    dict(id='H_air', setup='horizontal', core=True, name='공중 닦기',
         cue='레인 위 2~3 cm 높이에서 닦는 동작만 똑같이. 표면에 닿지 않게, 팔은 편하게.'),
    dict(id='H_light', setup='horizontal', core=True, name='가볍게 닦기 (기준)',
         cue='표면에 살짝 닿을 만큼만 누르고 팔 힘을 뺀 채 편하게 닦기. 경로 정확도는 신경 쓰지 않기.'),
    dict(id='H_stiff_z', setup='horizontal', core=True, name='z 강성: 누르는 방향 단단히',
         cue='팔을 굳혀 수직 방향을 단단히: 일정하고 강한 힘으로 꾹 누른 채 닦기, 도구가 뜨지 않게 버티기. 좌우 경로는 대충.'),
    dict(id='H_stiff_xy', setup='horizontal', core=True, name='xy 강성: 표면 방향 단단히',
         cue='표면 방향을 단단히: 레인을 벗어나지 않게 정확히, 흔들리지 않게 따라가기. 누르는 힘은 가볍게(H_light 수준).'),
    dict(id='H_stiff_all', setup='horizontal', core=True, name='전체 강성',
         cue='팔 전체를 최대한 단단히 굳힌 채(동시 수축): 꾹 누르면서 레인도 정확히. 아프지 않은 범위.'),
    dict(id='V_light', setup='vertical', core=True, name='수직 가볍게 닦기',
         cue='벽에 살짝 닿을 만큼만 밀고 편하게 닦기. 경로 정확도는 신경 쓰지 않기.'),
    dict(id='V_stiff_z', setup='vertical', core=True, name='수직 z 강성: 벽 쪽 단단히',
         cue='벽을 향한 방향을 단단히: 강하고 일정하게 벽을 민 채 닦기. 좌우 경로는 대충.'),
    dict(id='V_stiff_xy', setup='vertical', core=True, name='수직 xy 강성: 벽면 방향 단단히',
         cue='벽면 방향을 단단히: 레인을 정확히 따라가기. 미는 힘은 가볍게(V_light 수준).'),
    dict(id='V_air', setup='vertical', core=False, name='수직 공중 닦기',
         cue='벽 앞 2~3 cm에서 닦는 동작만. 닿지 않게.'),
    dict(id='V_stiff_all', setup='vertical', core=False, name='수직 전체 강성',
         cue='팔 전체를 최대한 단단히: 강하게 밀면서 레인도 정확히. 아프지 않은 범위.'),
    dict(id='H_press', setup='horizontal', core=False, name='누르기만 (정지)',
         cue='H_stiff_z와 같은 힘으로 레인 가운데를 누르기만, 움직이지 않기.'),
    dict(id='H_light_y', setup='horizontal', core=False, name='앞뒤 방향 가볍게 닦기',
         cue='H_light와 같지만 몸 쪽 <-> 바깥 방향(앞뒤)으로 닦기.'),
]
BY_ID = {s['id']: s for s in SCENARIOS}
# (A, B, question): AUC = P(trial of A > trial of B), per band and muscle group.
CONTRASTS = [('H_stiff_z', 'H_stiff_xy', '수평: z 강성 vs xy 강성'), ('V_stiff_z', 'V_stiff_xy', '수직: z 강성 vs xy 강성'),
             ('H_light', 'H_air', '수평: 실제 닦기 vs 공중'), ('V_light', 'V_air', '수직: 실제 닦기 vs 공중'),
             ('V_light', 'H_light', '가볍게: 수직 vs 수평'), ('V_stiff_z', 'H_stiff_z', 'z 강성: 수직 vs 수평'),
             ('H_stiff_all', 'H_light', '수평: 전체 강성 vs 가볍게'), ('H_stiff_z', 'H_press', '수평: 누르며 닦기 vs 누르기만'),
             ('H_light_y', 'H_light', '수평 가볍게: 앞뒤 vs 좌우')]
TASK_TRIM_S = (1., .5)   # task windows used for the summary: drop the first 1 s and the last 0.5 s


def make_plan(ids, trials, order='interleaved', seed=2026):
    if order == 'blocked':return [dict(scenario=s, trial=k) for s in ids for k in range(1, trials+1)]
    rng, plan = np.random.default_rng(seed), []
    for setup in dict.fromkeys(BY_ID[s]['setup'] for s in ids):
        group = [s for s in ids if BY_ID[s]['setup'] == setup]
        for k in range(1, trials+1):plan += [dict(scenario=group[i], trial=k) for i in rng.permutation(len(group))]
    return plan


def analyse_trial(directory, bands_map=None):
    """Median task RMS per band; with a map also the fraction of the mapping response per channel and muscle group."""
    p = Path(directory);meta = json.loads((p/'metadata.json').read_text());out = {}
    for role in meta['bands']:
        w = [r for v in phase_windows(p/role, 'task', TASK_TRIM_S[0], TASK_TRIM_S[1]).values() for r in v]
        pre = [r['rms'] for v in phase_windows(p/role, 'pre', .5).values() for r in v]
        band = (bands_map or {}).get(role)
        res = dict(windows=len(w), rms=None, norm=None, groups={}, level=None, pre_level=None, hum_max=None, imu=None)
        if w:
            rms = np.median([r['rms'] for r in w], 0)
            res.update(rms=rms.tolist(), hum_max=float(np.max(np.median([r['harmonic_fraction'] for r in w], 0))))
            if band:
                n = normalize(rms, band)
                res.update(norm=n['norm'], level=n['level'], groups=n['groups'],
                           pre_level=normalize(np.median(pre, 0), band)['level'] if pre else None)
            res['imu'] = task_imu(p/role, {r['event_id'] for r in w}, min(r['end_time_s'] for r in w)-.1, max(r['end_time_s'] for r in w))
        out[role] = res
    result = dict(scenario=meta['scenario']['id'], trial=meta['trial'], status=meta['status'], bands=out,
                  task_window_s=f'task phase without its first {TASK_TRIM_S[0]} s and last {TASK_TRIM_S[1]} s',
                  note='norm = (RMS - mapping rest) / (mapping task response - rest); 1 = as strong as the strongest mapping task, not MVC.')
    save_json(p/'analysis.json', result)
    return result


def run_trial(a, exp, exp_dir, index, bands_map):
    step = exp['plan'][index];sc = BY_ID[step['scenario']]
    name = f"{index+1:03d}_{sc['id']}_t{step['trial']:02d}";d = exp_dir/'trials'/name;k = 1
    while d.exists():k += 1;d = exp_dir/'trials'/f'{name}_a{k}'
    label = exp['scenarios'].index(sc['id'])+1
    group = Bands(MapRecorder, d, SimpleNamespace(port=None, baud=a.baud, rate=a.rate, simulate=a.simulate), a.bands)
    meta = dict(kind='stiffness_direction_trial', status='recording', simulated=a.simulate, rate=a.rate, baud=a.baud, filter=filter_settings(a.rate),
                scenario=sc, trial=step['trial'], plan_index=index, label=label, durations_s=exp['durations_s'], pace_s=exp['pace_s'],
                mapping_dir=exp.get('mapping_dir'), experiment=str(exp_dir.resolve()), note=exp['note'],
                label_semantics='label = 1 + index in experiment.json scenarios; round = trial number; phases pre -> task -> post')
    group.save_meta(meta);group.start()
    s = exp['durations_s']
    try:
        wait_for_data(group)
        timed(group, 'pre', step['trial'], label, s['pre'], '  준비: 도구 잡고 시작 위치에서 가만히', bands_map)
        beep(a.quiet, 'Glass')
        timed(group, 'task', step['trial'], label, s['task'], f"  >> {sc['name']}", bands_map, exp['pace_s'], a.quiet)
        beep(a.quiet, 'Glass')
        timed(group, 'post', step['trial'], label, s['post'], '  멈추고 힘 빼기', bands_map)
        meta['status'] = 'completed'
    except KeyboardInterrupt:
        meta['status'] = 'aborted';print('\n  trial 중단')
    except RuntimeError as e:
        meta.update(status='error', error=str(e));print('\n  error:', e)
    finally:
        group.set_phase('end');group.stop();ss = group.snapshot()
        meta.update(status='error' if ss['error'] else meta['status'], error=ss['error'] or meta.get('error'),
                    counter_breaks=ss['breaks'], discarded_bytes=group.discarded())
        group.save_meta(meta)
    entry = dict(index=index, scenario=sc['id'], trial=step['trial'], dir=str(d.relative_to(exp_dir)), status=meta['status'], note='')
    if meta['status'] == 'completed':
        r = analyse_trial(d, bands_map)
        entry['summary'] = {role: dict(level=b['level'], groups=b['groups']) for role, b in r['bands'].items()}
        print('  '+' | '.join(f"{role} {fmt(b['level'])} ("+', '.join(f'{g} {fmt(v)}' for g, v in b['groups'].items())+')' for role, b in r['bands'].items()))
    return entry


def fmt(x):
    return '--' if x is None else f'{x:.2f}'


def analyse_experiment(exp_dir):
    """trials.csv (one row per valid trial and band), summary.json/.md and summary.png from all completed, not redone trials."""
    exp_dir = Path(exp_dir);exp = json.loads((exp_dir/'experiment.json').read_text())
    bands_map = map_of(exp_dir, exp)
    rows = []
    for e in exp['entries']:
        if e['status'] != 'completed' or e.get('discarded'):continue
        r = analyse_trial(exp_dir/e['dir'], bands_map)
        for role, b in r['bands'].items():
            if b['rms'] is None:continue
            rows.append(dict(scenario=e['scenario'], setup=BY_ID[e['scenario']]['setup'], trial=e['trial'], band=role, dir=e['dir'], windows=b['windows'],
                             level=b['level'], pre_level=b['pre_level'], hum_max=b['hum_max'],
                             gyro_rms_rad_s=b['imu']['gyro_rms_rad_s'] if b['imu'] else None,
                             **{f'acc_{k}': b['imu']['acc_mean_m_s2'][i] if b['imu'] else None for i, k in enumerate('xyz')},
                             **{f'g_{g}': v for g, v in b['groups'].items()},
                             **{f'norm{i+1}': v for i, v in enumerate(b['norm'] or [None]*8)}, **{f'rms{i+1}': v for i, v in enumerate(b['rms'])},
                             note=e.get('note', '')))
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (exp_dir/'trials.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(rows)
    normed = bool(bands_map)
    summary, contrasts = {}, []
    for role in dict.fromkeys(r['band'] for r in rows):
        rr = [r for r in rows if r['band'] == role]
        keys = (['level']+[k for k in dict.fromkeys(k for r in rr for k in r) if k.startswith('g_')]) if normed else [f'rms{i+1}' for i in range(8)]
        summary[role] = {s: dict(n=len(sel), **{k: float(np.median([r[k] for r in sel])) for k in keys if all(r.get(k) is not None for r in sel)})
                         for s in exp['scenarios'] if (sel := [r for r in rr if r['scenario'] == s])}
        for x, y, question in CONTRASTS:
            ax, ay = [r for r in rr if r['scenario'] == x], [r for r in rr if r['scenario'] == y]
            if len(ax) < 2 or len(ay) < 2:continue
            for k in keys:
                if all(r.get(k) is not None for r in ax+ay):
                    contrasts.append(dict(band=role, a=x, b=y, question=question, measure=k, auc=auc([r[k] for r in ax], [r[k] for r in ay]),
                                          median_a=float(np.median([r[k] for r in ax])), median_b=float(np.median([r[k] for r in ay]))))
    result = dict(experiment=str(exp_dir.resolve()), valid_trials=len({r['dir'] for r in rows}), planned=len(exp['plan']), normalized=normed,
                  summary=summary, contrasts=contrasts,
                  note=('values = median over trials of the fraction of the mapping response (1 = strongest mapping task, not MVC)' if normed
                        else 'no muscle map: values = median filtered RMS counts'))
    save_json(exp_dir/'summary.json', result)
    (exp_dir/'summary.md').write_text(summary_md(result))
    plot_summary(rows, exp['scenarios'], exp_dir/'summary.png', normed)
    return result


def summary_md(result):
    lines = ['# Stiffness-direction demo summary', '', f"{result['valid_trials']} valid trials (planned {result['planned']}). {result['note']}.", '']
    for role, sc in result['summary'].items():
        keys = list(dict.fromkeys(k for v in sc.values() for k in v if k != 'n'))
        lines += [f'## {role}', '', '| scenario | n | '+' | '.join(k.removeprefix('g_') for k in keys)+' |', '|---|---|'+'---|'*len(keys)]
        lines += [f"| {s} | {v['n']} | "+' | '.join(fmt(v.get(k)) for k in keys)+' |' for s, v in sc.items()]
        lines.append('')
    if result['contrasts']:
        lines += ['## Contrasts', '', 'AUC = P(trial of A > trial of B); 0.5 = no difference, >= 0.8 or <= 0.2 = consistent across trials.', '',
                  '| question | band | measure | A | B | AUC |', '|---|---|---|---|---|---|']
        lines += [f"| {c['question']} | {c['band']} | {c['measure'].removeprefix('g_')} | {fmt(c['median_a'])} | {fmt(c['median_b'])} | {c['auc']:.2f} |"
                  for c in result['contrasts']]
    return '\n'.join(lines)+'\n'


def plot_summary(rows, scenarios, path, normed):
    if not rows:return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    roles = list(dict.fromkeys(r['band'] for r in rows));scen = [s for s in scenarios if any(r['scenario'] == s for r in rows)]
    key = 'norm' if normed else 'rms'
    m = np.full((len(scen), 8*len(roles)), np.nan)
    for i, s in enumerate(scen):
        for j, role in enumerate(roles):
            sel = [r for r in rows if r['scenario'] == s and r['band'] == role]
            for c in range(8):
                v = [r[f'{key}{c+1}'] for r in sel if r.get(f'{key}{c+1}') is not None]
                if v:m[i, 8*j+c] = np.median(v)
    fig, ax = plt.subplots(figsize=(1.2+.45*m.shape[1], 1.2+.4*len(scen)))
    im = ax.imshow(m, cmap='viridis', aspect='auto', vmin=0, vmax=np.nanmax(m) if normed and np.isfinite(m).any() else None)
    ax.set_xticks(range(m.shape[1]), [f'{role[0].upper()}{c+1}' for role in roles for c in range(8)])
    ax.set_yticks(range(len(scen)), scen)
    for j in range(1, len(roles)):ax.axvline(8*j-.5, color='w', lw=2)
    ax.set_title('median over trials: '+('fraction of mapping response' if normed else 'filtered RMS (counts)'))
    fig.colorbar(im, ax=ax, shrink=.8);fig.savefig(path, dpi=110, bbox_inches='tight');plt.close(fig)


def map_of(exp_dir, exp):
    """Muscle-map bands of an experiment, {} without a usable map."""
    p = exp_dir/exp['mapping_dir'] if exp.get('mapping_dir') else None
    return load_map(p) if p and (p/'muscle_map.json').exists() else {}


def next_index(exp):
    done = {e['index'] for e in exp['entries'] if e['status'] == 'completed' and not e.get('discarded')}
    return next((i for i in range(len(exp['plan'])) if i not in done), None)


def mapping_stage(a, exp, exp_dir):
    """Run (or redo) the muscle mapping until the operator accepts it; returns False if the operator quits."""
    while not exp.get('mapping_dir'):
        k = 1+sum(1 for _ in exp_dir.glob('mapping*'))
        d = exp_dir/('mapping' if k == 1 else f'mapping_{k}')
        run_mapping(a, d)
        line = ask('\nENTER: 이 매핑으로 시나리오 시작 | m: 매핑 다시 | q: 저장하고 종료 > ').lower()
        if line == 'q':return False
        if line != 'm':exp['mapping_dir'] = str(d.relative_to(exp_dir));save_json(exp_dir/'experiment.json', exp)
    return True


def run_experiment(a):
    if a.resume:
        exp_dir = Path(a.resume);exp = json.loads((exp_dir/'experiment.json').read_text())
        if set(exp['bands']) != set(a.bands):raise SystemExit(f"--band roles {list(a.bands)} differ from this experiment's {list(exp['bands'])}")
    else:
        exp_dir = session_dir(a.output, 'stiffness_direction')
        exp = dict(kind='stiffness_direction', created=datetime.now().isoformat(timespec='seconds'), simulated=a.simulate, rate=a.rate,
                   bands=a.bands, scenarios=a.scenarios, scenario_defs=[BY_ID[s] for s in a.scenarios], setups=SETUPS, common=COMMON,
                   trials_per_scenario=a.trials, order=a.order, seed=a.seed, durations_s=dict(pre=a.pre_s, task=a.task_s, post=a.post_s),
                   pace_s=a.pace_s, note=a.note, plan=make_plan(a.scenarios, a.trials, a.order, a.seed), mapping_dir=None, entries=[])
        if a.mapping:
            exp.update(mapping_dir=str(Path(a.mapping).resolve()), mapping_reused=True)
            print('기존 매핑 사용 (같은 착용 상태에서 만든 것이어야 합니다):', a.mapping)
        save_json(exp_dir/'experiment.json', exp)
    print('experiment:', exp_dir)
    if not mapping_stage(a, exp, exp_dir):return exp_dir
    bands_map = map_of(exp_dir, exp)
    if bands_map:print_mapping(json.loads((exp_dir/exp['mapping_dir']/'muscle_map.json').read_text()))
    print(f'\n=== 시나리오: {len(exp["scenarios"])}개 x {exp["trials_per_scenario"]}회 ({exp["order"]}) ===\n{COMMON}')
    last, setup = None, None
    while (i := next_index(exp)) is not None:
        step = exp['plan'][i];sc = BY_ID[step['scenario']]
        done = sum(1 for e in exp['entries'] if e['status'] == 'completed' and not e.get('discarded'))
        if sc['setup'] != setup:print(f"\n##### 셋업: {SETUPS[sc['setup']]}");setup = sc['setup']
        print(f"\n[{done+1}/{len(exp['plan'])}] {sc['id']} - {sc['name']} (trial {step['trial']}/{exp['trials_per_scenario']})\n  {sc['cue']}")
        line = ask('  ENTER: 시작 | r: 직전 trial 다시 | n 메모: 직전 trial에 메모 | q: 저장하고 종료 > ')
        if line.lower() == 'q':break
        if line.lower() == 'r':
            if last is None:print('  다시 할 직전 trial이 없습니다.');continue
            last['discarded'] = True;save_json(exp_dir/'experiment.json', exp);print(f"  {last['dir']} 버림 -> 다시 수행");last = None;continue
        if line[:2].lower() == 'n ':
            if last is None:print('  메모할 직전 trial이 없습니다.');continue
            last['note'] = (last['note']+'; ' if last['note'] else '')+line[2:].strip();save_json(exp_dir/'experiment.json', exp);continue
        entry = run_trial(a, exp, exp_dir, i, bands_map)
        exp['entries'].append(entry);save_json(exp_dir/'experiment.json', exp)
        if entry['status'] == 'completed':last = entry
        else:print('  저장되지 않은 trial: 같은 trial을 다시 시작합니다.')
    remaining = next_index(exp)
    print('\n모든 trial 완료.' if remaining is None else f'\n중단: 이어서 하려면 --resume {exp_dir}')
    result = analyse_experiment(exp_dir)
    print(f"{result['valid_trials']}/{result['planned']} valid trials | summary: {exp_dir/'summary.md'}")
    return exp_dir


def main(argv=None):
    p = argparse.ArgumentParser(prog='emg-probe direction', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--band', action='append', default=[], metavar='ROLE=PORT', help=f'ROLE in {list(ROLES)}; port optional with --simulate')
    p.add_argument('--simulate', action='store_true')
    p.add_argument('--baud', type=int, default=BAUD)
    p.add_argument('--rate', type=int, default=2000)
    p.add_argument('--scenarios', default='core', help="'core' (default), 'all', or comma-separated ids (see --list)")
    p.add_argument('--trials', type=int, default=10, help='Trials per scenario')
    p.add_argument('--order', choices=['interleaved', 'blocked'], default='interleaved')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--pre-s', type=float, default=2.)
    p.add_argument('--task-s', type=float, default=8.)
    p.add_argument('--post-s', type=float, default=1.)
    p.add_argument('--pace-s', type=float, default=1., help='Stroke pace sound period during the task; 0 = off')
    p.add_argument('--quiet', action='store_true', help='No sounds')
    p.add_argument('--map-reps', type=int, default=3)
    p.add_argument('--mapping', type=Path, help='Reuse a mapping folder (muscle_map.json) made in this donning instead of mapping first')
    p.add_argument('--resume', type=Path, help='Continue an experiment folder')
    p.add_argument('--analyse', type=Path, help='Recompute trials.csv / summary for an experiment folder')
    p.add_argument('--list', action='store_true', help='Print scenarios and exit')
    p.add_argument('--note', default='')
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    a = p.parse_args(argv)
    if a.list:
        for s in SCENARIOS:print(f"{s['id']:<12} {'core' if s['core'] else 'opt ':<5} {s['setup']:<10} {s['name']}: {s['cue']}")
        return
    if a.analyse:analyse_experiment(a.analyse);print((a.analyse/'summary.md').read_text());return
    try:a.bands = parse_bands(a.band, a.simulate)
    except ValueError as e:p.error(str(e))
    if not a.bands:p.error('Give --band wrist=PORT and/or --band upperarm=PORT')
    a.scenarios = ([s['id'] for s in SCENARIOS if s['core']] if a.scenarios == 'core' else [s['id'] for s in SCENARIOS] if a.scenarios == 'all'
                   else [x.strip() for x in a.scenarios.split(',') if x.strip()])
    if unknown := [s for s in a.scenarios if s not in BY_ID]:p.error(f'Unknown scenarios {unknown}; see --list')
    if len(set(a.scenarios)) != len(a.scenarios) or not a.scenarios:p.error('--scenarios needs distinct ids')
    if a.trials < 1 or a.map_reps < 2 or min(a.pre_s, a.task_s) <= 0 or a.task_s <= sum(TASK_TRIM_S):p.error('Check --trials/--map-reps/durations')
    if a.mapping:
        if not (a.mapping/'muscle_map.json').exists():p.error(f'{a.mapping} has no muscle_map.json')
        try:load_map(a.mapping, a.rate, a.simulate)
        except ValueError as e:p.error(str(e))
    run_experiment(a)


if __name__ == '__main__':
    main()
