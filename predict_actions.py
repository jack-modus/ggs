"""
predict_actions.py  –  GitHub Actions version of predict_v4.py.

Uses lookups_v6.pkl instead of races_master_v4.csv (17MB vs 272MB).
Outputs picks as both terminal text and picks.html for GitHub Pages.

Usage: python predict_actions.py <racecard.csv>
"""

import pandas as pd
import numpy as np
import re, pickle, sys, warnings, os
from datetime import datetime
warnings.filterwarnings('ignore')

RACECARD = sys.argv[1] if len(sys.argv) > 1 else exit("Usage: predict_actions.py <racecard.csv>")

print("Loading models and lookups...")
with open('model_bundle_v6_flat.pkl',  'rb') as f: FB = pickle.load(f)
with open('model_bundle_v6_jumps.pkl', 'rb') as f: JB = pickle.load(f)
with open('lookups_v6.pkl', 'rb') as f: L = pickle.load(f)

going_pref_lu   = L['going_pref_lu']
flat_sf_lu      = L['flat_sf_lu']
jumps_sf_lu     = L['jumps_sf_lu']
t_ov            = L['t_ov']
j_ov            = L['j_ov']
t_co            = L['t_co']
j_co            = L['j_co']
draw_lu         = L['draw_lu']
draw_lu_c       = L['draw_lu_c']
last_comment_lu = L['last_comment_lu']
t14             = L['t14']
j14             = L['j14']
prev_lu         = L['prev_lu']
career_runs_lu  = L['career_runs_lu']
pop_wr          = L['pop_wr']
pop_pl          = L['pop_pl']

def going_grp(g):
    g = str(g)
    if any(x in g for x in ['Firm','Fast','Hard']): return 'Fast'
    if 'Good to Firm' in g: return 'GoodFirm'
    if g=='Good': return 'Good'
    if 'Good to Soft' in g: return 'GoodSoft'
    if g=='Soft': return 'Soft'
    if 'Heavy' in g: return 'Heavy'
    if any(x in g for x in ['Standard','Slow','Polytrack']): return 'AW'
    return 'Other'

print(f"Loading racecard: {RACECARD}")
rc = pd.read_csv(RACECARD)
rc.columns = rc.columns.str.strip()
if 'Horse' in rc.columns: rc = rc.rename(columns={'Horse':'HorseName'})
if 'Run' in rc.columns and 'Ran' not in rc.columns:
    rc['Ran'] = pd.to_numeric(rc['Run'], errors='coerce')

def to_dec(x):
    if pd.isna(x) or str(x).strip() in ['','-']: return np.nan
    try:
        if '/' in str(x): n,d = str(x).split('/'); return float(n)/float(d)+1
        return float(x)+1
    except: return np.nan

rc['Decimal'] = rc['Odds'].apply(to_dec)
rc['FcstDec'] = rc['FcstOdds'].apply(to_dec) if 'FcstOdds' in rc.columns else np.nan
for col in ['OR','Age','Ran','Draw','WtLbs','Class']:
    if col in rc.columns: rc[col] = pd.to_numeric(rc[col], errors='coerce')
rc['CDB']     = rc['CDB'].fillna('').astype(str)
rc['Aid']     = rc['Aid'].fillna('').astype(str).str.strip()
rc['IsJump']  = rc['RaceDesc'].str.contains('Hurdle|Chase|Bumper|National Hunt',case=False,na=False)
rc['RaceKey'] = rc['Course'] + '|' + rc['Time'].astype(str)
rc['GoingGrp']= rc['Going'].apply(going_grp)
print(f"  {len(rc)} runners across {rc['RaceKey'].nunique()} races")

def parse_form(s, n=3):
    if pd.isna(s): return []
    s = str(s)
    for i in range(14,9,-1): s = s.replace(str(i),chr(64+i))
    nums=[int(d) if d.isdigit() else ord(d)-64 for d in re.findall(r'[\dA-N]',s)]
    return nums[-n:] if nums else []

def parse_days(x):
    if pd.isna(x): return np.nan
    m = re.match(r'[\d.]+', str(x).strip())
    return float(m.group()) if m else np.nan

def days_score(d, is_jump):
    if pd.isna(d): return 0.5
    if not is_jump:
        if d<=7: return 1.0
        if d<=21: return 0.80
        if d<=35: return 0.75
        if d<=60: return 0.65
        if d<=90: return 0.55
        return 0.45
    else:
        if d<=7:   return 0.85
        if d<=35:  return 0.80
        if d<=60:  return 0.75
        if d<=90:  return 0.65
        if d<=180: return 0.55
        return 0.40

def market_place_prob(dec, ran):
    try:
        frac = 5.0 if ran>=8 else (4.0 if ran>=5 else None)
        return 1/((dec-1)/frac+1) if frac else np.nan
    except: return np.nan

def build_features(r, race_df, bundle, is_jump, sf_lu_df):
    B    = bundle
    meds = B['medians']
    horse  = r['HorseName']
    course = r['Course']

    # Prev run from lookup
    pr = prev_lu[prev_lu['HorseName']==horse]
    prev_fpos  = pr['PrevFPos'].values[0]  if len(pr) else np.nan
    prev_dec   = pr['PrevDec'].values[0]   if len(pr) else np.nan
    prev_or    = pr['PrevOR'].values[0]    if len(pr) else np.nan
    prev_class = pr['PrevClass'].values[0] if len(pr) else np.nan
    prev_aid   = str(pr['Aid'].values[0])  if len(pr) else ''
    prev_spd   = pr['PrevSpDrift'].values[0] if len(pr) else np.nan
    prev_steam = float(prev_spd < -15) if pd.notna(prev_spd) else 0.0

    # Going pref
    gp_row = going_pref_lu[(going_pref_lu['HorseName']==horse) &
                            (going_pref_lu['GoingGrp']==r['GoingGrp'])]
    going_pref    = gp_row['GoingPref'].values[0]      if len(gp_row) else 0.0
    horse_overall = gp_row['HorseOverallWR'].values[0] if len(gp_row) else pop_wr

    # SF
    sf_row  = sf_lu_df[sf_lu_df['HorseName']==horse]
    best_sf = sf_row['BestSF3'].values[0] if len(sf_row) and pd.notna(sf_row['BestSF3'].values[0]) else np.nan

    or_val   = r.get('OR', np.nan)
    or_max   = race_df['OR'].max()
    or_rank  = race_df['OR'].rank(ascending=False, method='min').get(r.name, meds.get('ORRank',5))
    or_diff  = float(np.clip((or_val - or_max), -30, 0)) if pd.notna(or_val) else 0.0
    or_chg   = float(np.clip(or_val - prev_or, -20, 20)) if pd.notna(or_val) and pd.notna(prev_or) else 0.0
    wt_rank  = race_df['WtLbs'].rank(ascending=False, method='min').get(r.name, meds.get('WtRank',5)) if 'WtLbs' in race_df.columns else meds.get('WtRank',5)

    draw_n   = r.get('Draw', np.nan)
    ran      = r.get('Ran', 10) or 10
    draw_pct = draw_n/ran if pd.notna(draw_n) and ran>0 else np.nan
    dist     = r.get('Dist','') or r.get('Distance','')
    def _dbias(c, dist, dp, n):
        if pd.isna(dp) or n<8: return 1.0
        lut = draw_lu.get((c,dist), draw_lu_c.get(c,{}))
        if dp<=0.333: return lut.get('low',1.0)
        if dp>=0.667: return lut.get('high',1.0)
        return 1.0
    draw_bias = _dbias(course, dist, draw_pct, ran)

    fl = parse_form(r.get('Form'), 3)
    last_pos   = fl[-1]     if fl else np.nan
    avg_last3  = np.mean(fl) if fl else np.nan
    best_last3 = min(fl)     if fl else np.nan

    def _lu(df, key, col, fallback):
        row = df[df[key]==r.get(key,'')]
        return row[col].values[0] if len(row) and pd.notna(row[col].values[0]) else fallback
    def _lu2(df, k1, k2, col, fallback, v1):
        row = df[(df[k1]==v1) & (df[k2]==course)]
        return row[col].values[0] if len(row) and pd.notna(row[col].values[0]) else fallback

    t_wr  = _lu(t_ov, 'Trainer', 't_wr',  pop_wr)
    t_pr  = _lu(t_ov, 'Trainer', 't_pr',  pop_pl)
    j_wr  = _lu(j_ov, 'Jockey',  'j_wr',  pop_wr)
    j_pr  = _lu(j_ov, 'Jockey',  'j_pr',  pop_pl)
    tc_wr = _lu2(t_co,'Trainer','Course','tc_wr', t_wr, r.get('Trainer',''))
    tc_pr = _lu2(t_co,'Trainer','Course','tc_pr', t_pr, r.get('Trainer',''))
    jc_wr = _lu2(j_co,'Jockey', 'Course','jc_wr', j_wr, r.get('Jockey',''))
    jc_pr = _lu2(j_co,'Jockey', 'Course','jc_pr', j_pr, r.get('Jockey',''))

    t14_row = t14[t14['Trainer']==r.get('Trainer','')]
    j14_row = j14[j14['Jockey'] ==r.get('Jockey','')]
    trainer_wr_14d = t14_row['trainer_wr_14d'].values[0] if len(t14_row) else np.nan
    jockey_wr_14d  = j14_row['jockey_wr_14d'].values[0]  if len(j14_row) else np.nan

    cm_row = last_comment_lu[last_comment_lu['HorseName']==horse]
    def _cm(col): return float(cm_row[col].values[0]) if len(cm_row) and pd.notna(cm_row[col].values[0]) else 0.0
    last_won_well  = _cm('last_comment_won_well')
    last_kept_on   = _cm('last_comment_kept_on')
    last_weakened  = _cm('last_comment_weakened')
    last_outpaced  = _cm('last_comment_outpaced')
    last_promising = _cm('last_comment_promising')

    class_num   = r.get('Class', np.nan)
    class_delta = float(prev_class - class_num) if pd.notna(prev_class) and pd.notna(class_num) else 0.0
    class_drop  = float(class_delta >= 1)
    class_rise  = float(class_delta <= -1)

    cdb    = str(r.get('CDB',''))
    has_cd = float('CD' in cdb)
    has_c  = float(bool(re.search(r'(?<![C])C(?!D)',cdb)))
    has_d  = float('D' in cdb)
    has_bf = float('bf' in cdb.lower())

    curr_g = set(re.findall(r'[a-z]+', str(r.get('Aid','')).lower()))
    prev_g = set(re.findall(r'[a-z]+', prev_aid.lower()))
    if curr_g and not prev_g: hg = 1.0
    elif not curr_g and prev_g: hg = -0.5
    elif curr_g != prev_g: hg = 0.3
    else: hg = 0.0

    is_bf = float(pd.notna(prev_fpos) and prev_fpos>1 and prev_fpos<=4 and (prev_dec or 99)<=3.5)
    ds    = days_score(parse_days(r.get('Days')), is_jump)

    feat = {
        'tc_wr': tc_wr, 'tc_pr': tc_pr, 'jc_wr': jc_wr, 'jc_pr': jc_pr,
        't_wr': t_wr,   't_pr': t_pr,   'j_wr': j_wr,   'j_pr': j_pr,
        'trainer_wr_14d': trainer_wr_14d if pd.notna(trainer_wr_14d) else meds.get('trainer_wr_14d', pop_wr),
        'jockey_wr_14d':  jockey_wr_14d  if pd.notna(jockey_wr_14d)  else meds.get('jockey_wr_14d', pop_wr),
        'PrevSteamed': prev_steam,
        'ORRank': or_rank, 'ORDiffTop': or_diff, 'ORChange': or_chg,
        'SFRank': meds.get('SFRank',5), 'SFDiff': 0.0,
        'BestSF3': best_sf if pd.notna(best_sf) else meds.get('BestSF3',0),
        'LastPos': last_pos, 'AvgLast3': avg_last3, 'BestLast3': best_last3,
        'GoingPref': going_pref, 'HorseOverallWR': horse_overall,
        'DrawBias': draw_bias,
        'DrawPct': draw_pct if pd.notna(draw_pct) else meds.get('DrawPct',0.5),
        'DaysScore': ds, 'Ran': ran,
        'Age': r.get('Age', meds.get('Age',5)), 'WtRank': wt_rank,
        'HasBF': has_bf, 'HasCD': has_cd, 'HasC': has_c, 'HasD': has_d,
        'HGScore': hg, 'ClassDrop': class_drop, 'ClassRise': class_rise, 'IsBeatenFav': is_bf,
        'class_delta': class_delta,
        'last_comment_won_well': last_won_well, 'last_comment_kept_on': last_kept_on,
        'last_comment_weakened': last_weakened,  'last_comment_outpaced': last_outpaced,
        'last_comment_green': 0.0, 'last_comment_hampered': 0.0,
        'last_comment_promising': last_promising,
    }
    if is_jump:
        feat['StepUp'] = 0.0; feat['IsBumper'] = 0.0

    extras = {
        'trainer_wr_14d': trainer_wr_14d, 'jockey_wr_14d': jockey_wr_14d,
        'last_won_well': last_won_well, 'last_kept_on': last_kept_on,
        'last_promising': last_promising, 'going_pref': going_pref,
        'has_cd': has_cd, 'has_c': has_c, 'tc_wr': tc_wr, 'jc_wr': jc_wr,
        'or_chg': or_chg,
    }
    return feat, best_sf, extras

print("Scoring runners...")
results = []
for race_key, race in rc.groupby('RaceKey'):
    is_jump  = bool(race['IsJump'].iloc[0])
    bundle   = JB if is_jump else FB
    sf_lu_df = jumps_sf_lu if is_jump else flat_sf_lu
    FEATS    = bundle['FEATURES']
    meds     = bundle['medians']

    feat_dicts, sf_vals, extras_list = [], [], []
    for idx, r in race.iterrows():
        fd, sf_val, ex = build_features(r, race, bundle, is_jump, sf_lu_df)
        feat_dicts.append((idx, r, fd))
        sf_vals.append(sf_val)
        extras_list.append(ex)

    sf_arr   = np.array([v if pd.notna(v) else np.nan for v in sf_vals], dtype=float)
    sf_ranks = pd.Series(sf_arr).rank(ascending=False, method='min').values
    sf_max   = np.nanmax(sf_arr) if not np.all(np.isnan(sf_arr)) else 0
    sf_diffs = np.where(np.isnan(sf_arr), 0, np.clip(sf_arr - sf_max, -30, 0))

    feat_rows = []
    for i, (idx, r, fd) in enumerate(feat_dicts):
        fd['SFRank'] = float(sf_ranks[i]) if not np.isnan(sf_ranks[i]) else meds.get('SFRank',5)
        fd['SFDiff'] = float(sf_diffs[i])
        row = [fd.get(f, meds.get(f,0)) for f in FEATS]
        row = [0.0 if (v is None or (isinstance(v,float) and np.isnan(v))) else v for v in row]
        feat_rows.append(row)

    X = np.array(feat_rows, dtype=float)
    raw_win   = bundle['win_model'].predict_proba(X)[:,1]
    raw_place = bundle['place_model'].predict_proba(X)[:,1]
    norm_win   = raw_win   / raw_win.sum()   if raw_win.sum()>0   else raw_win
    norm_place = raw_place / raw_place.sum() if raw_place.sum()>0 else raw_place

    for i, (idx, r, fd) in enumerate(feat_dicts):
        sf_r = float(sf_ranks[i]) if not np.isnan(sf_ranks[i]) else 99
        results.append({
            'RaceKey': race_key, 'Time': r['Time'], 'Course': r['Course'],
            'RaceDesc': r.get('RaceDesc',''), 'Going': r['Going'],
            'IsJump': is_jump, 'HorseName': r['HorseName'],
            'Trainer': r.get('Trainer',''), 'Jockey': r.get('Jockey',''),
            'OR': r.get('OR'), 'Odds': r.get('Odds'), 'FcstOdds': r.get('FcstOdds',''),
            'Decimal': r.get('Decimal'), 'FcstDec': r.get('FcstDec', np.nan),
            'Ran': r.get('Ran'), 'CDB': r.get('CDB',''), 'Form': r.get('Form',''),
            'SFRank': sf_r, 'GoingPref': fd['GoingPref'], 'tc_wr': fd['tc_wr'],
            'jc_wr': fd['jc_wr'], 'ORChange': fd['ORChange'],
            'trainer_wr_14d': extras_list[i]['trainer_wr_14d'],
            'jockey_wr_14d':  extras_list[i]['jockey_wr_14d'],
            'last_won_well':  extras_list[i]['last_won_well'],
            'last_kept_on':   extras_list[i]['last_kept_on'],
            'last_promising': extras_list[i]['last_promising'],
            'model_win': norm_win[i], 'model_place': norm_place[i],
            'market_prob_raw': 1/r['Decimal'] if pd.notna(r.get('Decimal')) and r['Decimal']>1 else np.nan,
            'market_place': market_place_prob(r.get('Decimal',np.nan), r.get('Ran',0)),
        })

out = pd.DataFrame(results)
mkt_sum = out.groupby('RaceKey')['market_prob_raw'].transform('sum')
out['market_win'] = out['market_prob_raw'] / mkt_sum
out['win_edge']   = out['model_win'] - out['market_win']
out['place_edge'] = out['model_place'] - out['market_place']

def mkt_move_pct(row):
    fd = row.get('FcstDec'); d = row.get('Decimal')
    if pd.notna(fd) and pd.notna(d) and fd>1 and d>1: return (d-fd)/fd
    return np.nan
out['mkt_move'] = out.apply(mkt_move_pct, axis=1)

def conviction_score(r):
    score = 0; signals = []
    edge = r['win_edge'] if pd.notna(r['win_edge']) else 0
    dec  = r['Decimal']  if pd.notna(r['Decimal'])  else 0
    if edge >= 0.10: score += 1; signals.append(f'edge {edge:+.0%}')
    if edge >= 0.15: score += 1; signals.append('strong edge')
    if pd.notna(dec) and 5 <= dec <= 17: score += 1; signals.append(f'{r["Odds"]} (sweet spot)')
    mv = r.get('mkt_move', np.nan)
    if pd.notna(mv) and mv <= -0.10: score += 1; signals.append(f'steam {mv:.0%}')
    t14d = r.get('trainer_wr_14d', np.nan)
    if pd.notna(t14d) and t14d >= 0.15: score += 1; signals.append(f'trainer {t14d:.0%}/14d')
    j14d = r.get('jockey_wr_14d', np.nan)
    if pd.notna(j14d) and j14d >= 0.15: score += 1; signals.append(f'jockey {j14d:.0%}/14d')
    if r.get('last_won_well',0)==1 or r.get('last_promising',0)==1:
        score += 1; signals.append('ran well last time')
    elif r.get('last_kept_on',0)==1:
        score += 0.5; signals.append('kept on last time')
    cdb = str(r.get('CDB',''))
    if 'CD' in cdb: score += 1; signals.append('CD winner')
    elif re.search(r'(?<![C])C(?!D)', cdb): score += 0.5; signals.append('course winner')
    gp = r.get('GoingPref', 0)
    if pd.notna(gp) and gp > 0.05: score += 1; signals.append(f'going +{gp:.0%}')
    return score, signals

out['conviction'], out['signals'] = zip(*out.apply(conviction_score, axis=1))

def assign_tier(r):
    e = r['win_edge'] if pd.notna(r['win_edge']) else 0
    cv = r['conviction']
    if cv >= 5 and e >= 0.10: return 'CONVICTION'
    if cv >= 3 and e >= 0.08: return 'SELECT'
    if cv >= 2 and e >= 0.06: return 'WATCH'
    return ''
out['tier'] = out.apply(assign_tier, axis=1)

# ── Terminal output ────────────────────────────────────────────────────────────
from zoneinfo import ZoneInfo
now_london = datetime.now(ZoneInfo('UTC')).astimezone(ZoneInfo('Europe/London'))
date_str = now_london.strftime('%A %-d %B %Y') if os.name != 'nt' else now_london.strftime('%A %d %B %Y')
time_str = now_london.strftime('%H:%M')
print(f"\n{'='*70}")
print(f"  PICKS — {date_str}")
print(f"  Model v6 | Conviction system | +28.6% ROI backtest (Jun-Apr 2026)")
print(f"{'='*70}")

conviction_picks = []
for section, df in [('FLAT', out[~out['IsJump']]), ('JUMPS', out[out['IsJump']])]:
    section_picks = df[df['tier'].isin(['CONVICTION','SELECT','WATCH'])].sort_values(
        ['tier','conviction'], ascending=[True,False])
    if len(section_picks) == 0: continue
    print(f"\n  {section}")
    for tier in ['CONVICTION','SELECT','WATCH']:
        picks = section_picks[section_picks['tier']==tier]
        if len(picks)==0: continue
        stake = {'CONVICTION':'2pt win','SELECT':'1pt win','WATCH':'0.5pt win'}[tier]
        print(f"\n    {tier}  [{stake}]")
        for _, r in picks.iterrows():
            cr = career_runs_lu.get(r['HorseName'], 0)
            form_note = f" [{cr}r]" if cr < 20 else ""
            sigs = '  |  '.join(r['signals']) if isinstance(r['signals'], list) else ''
            print(f"      {str(r['Time']):<7} {r['Course']:<14} {r['HorseName']:<26} @ {str(r['Odds']):>7}  cv={r['conviction']:.1f}/8{form_note}")
            print(f"        {sigs}")
            if tier == 'CONVICTION':
                conviction_picks.append(r)

print(f"\n{'='*70}")

# ── HTML output for GitHub Pages ───────────────────────────────────────────────
html_rows = ''
for tier in ['CONVICTION','SELECT','WATCH']:
    picks = out[out['tier']==tier].sort_values('conviction', ascending=False)
    if len(picks)==0: continue
    tier_class = tier.lower()
    stake = {'CONVICTION':'2pt win','SELECT':'1pt win','WATCH':'0.5pt win'}[tier]
    html_rows += f'<div class="tier-block {tier_class}"><h2>{tier} <span class="stake">{stake}</span></h2>'
    for _, r in picks.iterrows():
        cr = career_runs_lu.get(r['HorseName'], 0)
        sigs = r['signals'] if isinstance(r['signals'], list) else []
        sig_html = ' &bull; '.join(sigs)
        cr_note  = f'<span class="cr">{cr}r</span>' if cr < 20 else ''
        html_rows += f'''
        <div class="pick">
          <div class="pick-header">
            <span class="time">{r["Time"]}</span>
            <span class="course">{r["Course"]}</span>
            <span class="horse">{r["HorseName"]}</span>{cr_note}
            <span class="odds">@ {r["Odds"]}</span>
            <span class="cv">cv {r["conviction"]:.1f}/8</span>
            <span class="edge">edge {r["win_edge"]:+.0%}</span>
          </div>
          <div class="signals">{sig_html}</div>
        </div>'''
    html_rows += '</div>'

html = f'''<title>Racing Picks — {date_str}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:720px;margin:0 auto;padding:16px;background:#0f0f13;color:#e8e8e8}}
  h1{{font-size:1.2rem;color:#aaa;margin-bottom:4px}}
  .updated{{font-size:.8rem;color:#555;margin-bottom:8px}}
  .odds-warning{{font-size:.75rem;color:#a06a2a;background:#2a2016;border-radius:6px;padding:8px 10px;margin-bottom:24px}}
  .tier-block{{margin-bottom:32px}}
  h2{{font-size:1rem;letter-spacing:.08em;margin:0 0 12px}}
  .conviction h2{{color:#f5c542}}
  .select h2{{color:#7ec8e3}}
  .watch h2{{color:#aaa}}
  .stake{{font-weight:400;font-size:.85rem;opacity:.7;margin-left:8px}}
  .pick{{background:#1a1a22;border-radius:8px;padding:12px 14px;margin-bottom:8px}}
  .conviction .pick{{border-left:3px solid #f5c542}}
  .select .pick{{border-left:3px solid #7ec8e3}}
  .watch .pick{{border-left:3px solid #555}}
  .pick-header{{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;margin-bottom:6px}}
  .time{{color:#888;font-size:.85rem;min-width:42px}}
  .course{{color:#aaa;font-size:.85rem;min-width:90px}}
  .horse{{font-weight:600;font-size:1rem;flex:1}}
  .odds{{color:#f5c542;font-weight:600}}
  .cv{{color:#888;font-size:.8rem}}
  .edge{{color:#7ec8e3;font-size:.8rem}}
  .cr{{background:#333;color:#aaa;font-size:.7rem;padding:1px 5px;border-radius:3px;margin-left:4px}}
  .signals{{font-size:.8rem;color:#888;line-height:1.5}}
  .footer{{margin-top:32px;font-size:.75rem;color:#444;border-top:1px solid #222;padding-top:12px}}
</style>
<h1>Racing Picks</h1>
<div class="updated">{date_str} &bull; updated {time_str} UK time &mdash; Model v6 &bull; +28.6% ROI backtest</div>
<div class="odds-warning">Odds shown are a snapshot from the last update — always check the live price before staking.</div>
{html_rows}
<div class="footer">Conviction system: 5+ signals required for top tier. Avoid &lt;3/1.
Backtest Jun 2025–Apr 2026 on unseen data.</div>'''

os.makedirs('docs', exist_ok=True)
with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Saved docs/index.html ({len(out)} runners scored)")
out.to_csv('predictions_today.csv', index=False)
