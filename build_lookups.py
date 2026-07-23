"""
build_lookups.py  –  Pre-compute all static lookups from races_master_v4.csv
into a single compressed file for use by GitHub Actions.

Outputs: lookups_v6.pkl  (~10-15MB)

Contains:
  - going_pref_lu    : per horse per going group going preference
  - horse_overall_lu : per horse overall win rate
  - best_sf3_lu      : per horse best SF3 from last 3 runs
  - t_ov, j_ov       : trainer/jockey overall win/place rates
  - t_co, j_co       : trainer/jockey per-course rates (shrunk)
  - draw_lu, draw_lu_c : draw bias tables
  - par_cdg, par_cd  : par times for speed figures
  - last_comment_lu  : per horse last-run comment flags
  - pop_wr, pop_pl   : population win/place rates
  - trainer_wr_14d   : trainer rolling 14d form (from most recent window)
  - jockey_wr_14d    : jockey rolling 14d form

Run this locally whenever races_master_v4.csv is updated.
"""

import pandas as pd
import numpy as np
import pickle, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SHRINK = 100

print("Loading races_master_v4.csv...")
master = pd.read_csv('races_master_v4.csv', encoding='utf-8-sig', low_memory=False)
for col in ['FPos','OR','Sp','Decimal','WeightLBS','Ran','Age','Yards','Class',
            'DstBtn','TotalBtn','Seconds','SpDrift',
            'last_comment_won_well','last_comment_kept_on','last_comment_weakened',
            'last_comment_outpaced','last_comment_green','last_comment_hampered',
            'last_comment_promising']:
    master[col] = pd.to_numeric(master[col], errors='coerce')
master['RaceDate'] = pd.to_datetime(master['RaceDate'], errors='coerce')
master['Winner']   = (master['FPos'] == 1).astype(int)
master['Placed']   = (master['FPos'] <= 3).astype(int)
master['Aid']      = master['Aid'].fillna('').astype(str).str.strip()
master['CDB']      = master['CDB'].fillna('').astype(str)
master['IsFlat']   = master['IsFlat'].astype(bool) if 'IsFlat' in master.columns else ~master['Type'].isin(['h','c','b'])
master['IsJump']   = master['IsJump'].astype(bool) if 'IsJump' in master.columns else master['Type'].isin(['h','c','b'])
master = master.sort_values(['HorseName','RaceDate','RaceTime']).reset_index(drop=True)
print(f"  {len(master):,} rows loaded")

def going_group(g):
    g = str(g)
    if any(x in g for x in ['Firm','Fast','Hard']): return 'Fast'
    if 'Good to Firm' in g: return 'GoodFirm'
    if g == 'Good': return 'Good'
    if 'Good to Soft' in g: return 'GoodSoft'
    if g == 'Soft': return 'Soft'
    if 'Heavy' in g: return 'Heavy'
    if any(x in g for x in ['Standard','Slow','Polytrack']): return 'AW'
    return 'Other'

master['GoingGrp'] = master['Going'].apply(going_group)
pop_wr = master['Winner'].mean()
pop_pl = master['Placed'].mean()

# ── Going preference ───────────────────────────────────────────────────────────
print("Computing going preference...")
gg = master.groupby(['HorseName','GoingGrp'])
master['g_runs'] = gg.cumcount()
master['g_wins'] = gg['Winner'].cumsum() - master['Winner']
ga = master.groupby('HorseName')
master['a_runs'] = ga.cumcount()
master['a_wins'] = ga['Winner'].cumsum() - master['Winner']
master['GoingPref']      = ((master['g_wins']/master['g_runs'].replace(0,np.nan)) -
                            (master['a_wins']/master['a_runs'].replace(0,np.nan))).fillna(0)
master['HorseOverallWR'] = (master['a_wins']/master['a_runs'].replace(0,np.nan)).fillna(pop_wr)

going_pref_lu = (master.sort_values('RaceDate')
                 .groupby(['HorseName','GoingGrp'])[['GoingPref','HorseOverallWR']]
                 .last().reset_index())
horse_overall_lu = (master.sort_values('RaceDate')
                    .groupby('HorseName')['HorseOverallWR'].last().reset_index())

# ── Speed figures (build separately for flat and jumps) ───────────────────────
print("Computing speed figures...")
def compute_sf_lu(df, par_cdg, par_cd):
    df = df.copy()
    df['Furlongs']      = df['Yards'] / 220
    df['SecsPerLength'] = (df['Furlongs'] / 25).clip(0.10, 0.35)
    df = df.merge(par_cdg, on=['Course','Distance','GoingGrp'], how='left')
    df = df.merge(par_cd.rename(columns={'par':'par_cd'}), on=['Course','Distance'], how='left')
    df['par']      = df['par'].fillna(df['par_cd'])
    df['race_sf']  = (df['par'] - df['Seconds']) * 10
    df['DstBtn_c'] = df['DstBtn'].fillna(df['TotalBtn']).fillna(0)
    df['horse_sf'] = df['race_sf'] - df['DstBtn_c'] * df['SecsPerLength'] * 10
    sf_g = df.sort_values(['HorseName','RaceDate']).groupby('HorseName')
    df['sf1'] = sf_g['horse_sf'].shift(1)
    df['sf2'] = sf_g['horse_sf'].shift(2)
    df['sf3'] = sf_g['horse_sf'].shift(3)
    df['BestSF3'] = df[['sf1','sf2','sf3']].max(axis=1)
    return df.sort_values('RaceDate').groupby('HorseName')['BestSF3'].last().reset_index()

# Load v6 bundles for par times
with open('model_bundle_v6_flat.pkl',  'rb') as f: FB = pickle.load(f)
with open('model_bundle_v6_jumps.pkl', 'rb') as f: JB = pickle.load(f)

flat_sf_lu  = compute_sf_lu(master[master['IsFlat']].copy(), FB['par_cdg'], FB['par_cd'])
jumps_sf_lu = compute_sf_lu(master[master['IsJump']].copy(), JB['par_cdg'], JB['par_cd'])

# ── Trainer/jockey stats ───────────────────────────────────────────────────────
print("Building trainer/jockey tables...")
t_ov = (master.groupby('Trainer').agg(r=('Winner','count'),w=('Winner','sum'),p=('Placed','sum'))
        .assign(t_wr=lambda x:x['w']/x['r'], t_pr=lambda x:x['p']/x['r'])
        .query('r>=20')[['t_wr','t_pr']].reset_index())
j_ov = (master.groupby('Jockey').agg(r=('Winner','count'),w=('Winner','sum'),p=('Placed','sum'))
        .assign(j_wr=lambda x:x['w']/x['r'], j_pr=lambda x:x['p']/x['r'])
        .query('r>=20')[['j_wr','j_pr']].reset_index())

t_co = (master.groupby(['Trainer','Course']).agg(tr=('Winner','count'),tw=('Winner','sum'),tp=('Placed','sum'))
        .reset_index().merge(t_ov[['Trainer','t_wr','t_pr']], on='Trainer', how='left'))
t_co['t_wr'] = t_co['t_wr'].fillna(pop_wr); t_co['t_pr'] = t_co['t_pr'].fillna(pop_pl)
t_co['tc_wr'] = (t_co['tw'] + SHRINK*t_co['t_wr']) / (t_co['tr'] + SHRINK)
t_co['tc_pr'] = (t_co['tp'] + SHRINK*t_co['t_pr']) / (t_co['tr'] + SHRINK)

j_co = (master.groupby(['Jockey','Course']).agg(jr=('Winner','count'),jw=('Winner','sum'),jp=('Placed','sum'))
        .reset_index().merge(j_ov[['Jockey','j_wr','j_pr']], on='Jockey', how='left'))
j_co['j_wr'] = j_co['j_wr'].fillna(pop_wr); j_co['j_pr'] = j_co['j_pr'].fillna(pop_pl)
j_co['jc_wr'] = (j_co['jw'] + SHRINK*j_co['j_wr']) / (j_co['jr'] + SHRINK)
j_co['jc_pr'] = (j_co['jp'] + SHRINK*j_co['j_pr']) / (j_co['jr'] + SHRINK)

# ── Draw bias ──────────────────────────────────────────────────────────────────
print("Building draw bias tables...")
draw_lu, draw_lu_c = {}, {}
for (c, dist), sub in master[master['Ran']>=8].groupby(['Course','Distance']):
    sub = sub.copy(); sub['Draw_n'] = pd.to_numeric(sub['Draw'], errors='coerce')
    sub = sub[sub['Draw_n'].notna() & (sub['Draw_n']>0)]
    if len(sub)<50: continue
    sub['dp'] = sub['Draw_n']/sub['Ran']; exp = 1/sub['Ran'].mean()
    draw_lu[(c,dist)] = {'low': sub[sub['dp']<=0.333]['Winner'].mean()/exp,
                         'high': sub[sub['dp']>=0.667]['Winner'].mean()/exp}
for c, sub in master[master['Ran']>=8].groupby('Course'):
    sub = sub.copy(); sub['Draw_n'] = pd.to_numeric(sub['Draw'], errors='coerce')
    sub = sub[sub['Draw_n'].notna() & (sub['Draw_n']>0)]
    if len(sub)<100: continue
    sub['dp'] = sub['Draw_n']/sub['Ran']; exp = 1/sub['Ran'].mean()
    draw_lu_c[c] = {'low': sub[sub['dp']<=0.333]['Winner'].mean()/exp,
                    'high': sub[sub['dp']>=0.667]['Winner'].mean()/exp}

# ── Last-run comment flags ─────────────────────────────────────────────────────
print("Building comment flag lookup...")
last_comment_lu = (master.sort_values('RaceDate')
                   .groupby('HorseName')[['last_comment_won_well','last_comment_kept_on',
                                          'last_comment_weakened','last_comment_outpaced',
                                          'last_comment_promising']].last().reset_index())

# ── Rolling 14-day form (most recent window) ───────────────────────────────────
print("Computing 14-day rolling form...")
max_date = master['RaceDate'].max()
recent   = master[master['RaceDate'] >= max_date - pd.Timedelta(days=14)]
t14 = (recent.groupby('Trainer').agg(runs14=('Winner','count'), wins14=('Winner','sum'))
       .assign(trainer_wr_14d=lambda x: x['wins14']/x['runs14'])
       .query('runs14 >= 3')[['trainer_wr_14d']].reset_index())
j14 = (recent.groupby('Jockey').agg(runs14=('Winner','count'), wins14=('Winner','sum'))
       .assign(jockey_wr_14d=lambda x: x['wins14']/x['runs14'])
       .query('runs14 >= 3')[['jockey_wr_14d']].reset_index())

# ── Last-run prev features per horse ──────────────────────────────────────────
print("Building prev-run lookup...")
master['PrevSpDrift'] = master.groupby('HorseName')['SpDrift'].shift(1)
master['PrevAid']     = master.groupby('HorseName')['Aid'].shift(1).fillna('')
master['PrevFPos']    = master.groupby('HorseName')['FPos'].shift(1)
master['PrevDec']     = master.groupby('HorseName')['Decimal'].shift(1)
master['PrevOR']      = master.groupby('HorseName')['OR'].shift(1)
master['PrevClass']   = master.groupby('HorseName')['Class'].shift(1)
prev_lu = (master.sort_values('RaceDate')
           .groupby('HorseName')[['PrevSpDrift','PrevAid','PrevFPos','PrevDec','PrevOR','PrevClass','Aid','OR','Class']]
           .last().reset_index())

# ── Career runs ───────────────────────────────────────────────────────────────
career_runs_lu = master.groupby('HorseName').size().to_dict()

# ── Pack and save ──────────────────────────────────────────────────────────────
lookups = dict(
    going_pref_lu=going_pref_lu,
    horse_overall_lu=horse_overall_lu,
    flat_sf_lu=flat_sf_lu,
    jumps_sf_lu=jumps_sf_lu,
    t_ov=t_ov, j_ov=j_ov,
    t_co=t_co, j_co=j_co,
    draw_lu=draw_lu, draw_lu_c=draw_lu_c,
    last_comment_lu=last_comment_lu,
    t14=t14, j14=j14,
    prev_lu=prev_lu,
    career_runs_lu=career_runs_lu,
    pop_wr=pop_wr, pop_pl=pop_pl,
    data_as_of=str(max_date.date()),
)

print("Saving lookups_v6.pkl...")
with open('lookups_v6.pkl', 'wb') as f:
    pickle.dump(lookups, f, protocol=4)

import os
size_mb = os.path.getsize('lookups_v6.pkl') / 1e6
print(f"Done. lookups_v6.pkl = {size_mb:.1f} MB")
print(f"Data as of: {max_date.date()}")
print(f"Horses: {len(going_pref_lu['HorseName'].unique()):,}")
print(f"Trainers: {len(t_ov):,} | Jockeys: {len(j_ov):,}")
