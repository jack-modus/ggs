"""
fetch_racecard_sl.py  –  Scrapes today's GB/IRE racecard from SportingLife.

No Betfair API needed — works from any IP including GitHub Actions.
Two-stage scrape: listing page (meetings+races+country flag) then one
request per race (rides = runners with odds, form, OR, headgear, etc).

Odds column = current SL/bookmaker odds (fractional). Used as the market
price proxy in place of Betfair exchange odds.

Usage: python fetch_racecard_sl.py [YYYY-MM-DD]
"""

import requests, re, json, sys, io, os, time
import pandas as pd
from datetime import date, datetime
from zoneinfo import ZoneInfo

LONDON = ZoneInfo('Europe/London')

def to_local_time(hhmm_str, race_date):
    """SL's JSON time field is in UTC; convert to Europe/London local (BST/GMT-aware)."""
    m = re.match(r'(\d{1,2}):(\d{2})', str(hhmm_str or ''))
    if not m:
        return str(hhmm_str or '')
    h, mi = int(m.group(1)), int(m.group(2))
    utc_dt = datetime(race_date.year, race_date.month, race_date.day, h, mi, tzinfo=ZoneInfo('UTC'))
    local_dt = utc_dt.astimezone(LONDON)
    return local_dt.strftime('%H:%M')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

target_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
date_str    = target_date.strftime('%Y-%m-%d')
date_label  = target_date.strftime('%y%m%d')
out_file    = f"{date_label}racecard.csv"

HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/124.0.0.0 Safari/537.36'),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-GB,en;q=0.9',
}

COURSE_ALIASES = {
    'CHEL': 'CHELTENHAM', 'LING': 'LINGFIELD', 'KEMP': 'KEMPTON',
    'NEWB': 'NEWBURY', 'YORK': 'YORK', 'DONC': 'DONCASTER',
    'HAYD': 'HAYDOCK', 'GOOD': 'GOODWOOD', 'NEWM': 'NEWMARKET',
    'LEIC': 'LEICESTER', 'NOTT': 'NOTTINGHAM', 'WIND': 'WINDSOR',
    'BRIG': 'BRIGHTON', 'EPSO': 'EPSOM', 'SALI': 'SALISBURY',
    'BATH': 'BATH', 'CARL': 'CARLISLE', 'CATT': 'CATTERICK',
    'CHES': 'CHESTER', 'CHEP': 'CHEPSTOW', 'EXET': 'EXETER',
    'FFOR': 'FFOS LAS', 'HAMI': 'HAMILTON', 'HERE': 'HEREFORD',
    'HUNT': 'HUNTINGDON', 'LEOP': 'LEOPARDSTOWN', 'LUDO': 'LUDLOW',
    'MUSS': 'MUSSELBURGH', 'NAAS': 'NAAS', 'NAVA': 'NAVAN',
    'NEWC': 'NEWCASTLE', 'PERT': 'PERTH', 'PONT': 'PONTEFRACT',
    'REDC': 'REDCAR', 'RIPON': 'RIPON', 'SANDOWN': 'SANDOWN',
    'SOUT': 'SOUTHWELL', 'THIR': 'THIRSK', 'WORC': 'WORCESTER',
    'WOLVER': 'WOLVERHAMPTON', 'YARM': 'YARMOUTH',
}
HEADGEAR_SYMBOLS = {
    'p': 'p', 'b': 'b', 'v': 'v', 't': 't', 'h': 'h', 'e': 'e',
    'hd': 'hd', 'ts': 'ts', 'pc': 'p',
}

def normalise_course(name):
    n = re.sub(r'\s*\(.*?\)', '', str(name)).strip().upper()
    n = re.sub(r'[^A-Z0-9 ]', '', n).strip()
    return COURSE_ALIASES.get(n[:4], n)

def weight_to_lbs(handicap_str):
    m = re.match(r'(\d+)-(\d+)', str(handicap_str or ''))
    if not m: return None
    return int(m.group(1))*14 + int(m.group(2))

def get_json(url):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        return None
    nd = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text, re.S)
    if not nd:
        return None
    return json.loads(nd.group(1))

# ── Stage 1: listing page ──────────────────────────────────────────────────────
print(f"Fetching SL racecard listing for {date_str}...")
listing_url = f"https://www.sportinglife.com/racing/racecards/{date_str}"
data = get_json(listing_url)
if data is None:
    print("ERROR: could not load/parse listing page"); sys.exit(1)

meetings = data.get('props',{}).get('pageProps',{}).get('meetings',[])
print(f"  {len(meetings)} meetings found")

race_jobs = []   # (course, going, race_id, race_name_hint)
for meeting in meetings:
    summary = meeting.get('meeting_summary', {})
    course_field = summary.get('course', '')
    if isinstance(course_field, dict):
        course_name = course_field.get('name', '')
        country_obj = course_field.get('country', {})
        country = (country_obj.get('long_name','') if isinstance(country_obj, dict)
                   else str(country_obj)).lower()
    else:
        course_name = str(course_field)
        country = ''

    GB_IRE_LONG = {'england','scotland','wales','eire','ireland','northern ireland'}
    if country and country not in GB_IRE_LONG:
        continue

    course = normalise_course(course_name)
    for race in meeting.get('races', []):
        rid = race.get('race_summary_reference', {}).get('id')
        if rid:
            race_jobs.append({
                'course': course,
                'going':  race.get('going') or summary.get('going','Unknown'),
                'race_id': rid,
            })

print(f"  {len(race_jobs)} GB/IRE races to fetch")

# ── Stage 2: per-race rides ─────────────────────────────────────────────────────
rows = []
for i, job in enumerate(race_jobs):
    url = f"https://www.sportinglife.com/racing/racecards/{date_str}/x/racecard/{job['race_id']}/x"
    try:
        rd = get_json(url)
    except Exception as e:
        print(f"  [{i+1}/{len(race_jobs)}] race {job['race_id']} fetch failed: {e}")
        continue
    if rd is None:
        continue

    race = rd.get('props',{}).get('pageProps',{}).get('race')
    if not race:
        continue
    rs = race.get('race_summary', {})

    course = job['course']
    going  = rs.get('going') or job['going'] or 'Unknown'
    race_time = to_local_time(rs.get('time',''), target_date)
    title  = rs.get('name','')
    dist   = rs.get('distance','')
    cls    = rs.get('race_class','')

    is_jump   = bool(re.search(r'hurdle|chase|bumper|national hunt', str(title), re.I))
    type_code = ('c' if 'chase' in str(title).lower() else
                 'b' if 'bumper' in str(title).lower() else
                 'h' if 'hurdle' in str(title).lower() else 'f')

    rides = race.get('rides', [])
    ran = len([r for r in rides if r.get('ride_status') == 'RUNNER'])
    if ran == 0:
        ran = len(rides)

    for ride in rides:
        if ride.get('ride_status') not in (None, 'RUNNER'):
            continue
        horse = ride.get('horse', {}) or {}
        horse_name = horse.get('name','')
        if not horse_name:
            continue

        trainer_name = (ride.get('trainer') or {}).get('name','')
        jockey_name  = (ride.get('jockey')  or {}).get('name','')
        form  = (horse.get('formsummary') or {}).get('display_text','')
        age   = horse.get('age','')
        days  = horse.get('last_ran_days','')
        draw  = ride.get('draw_number','')
        handicap = ride.get('handicap','')
        wt_lbs   = weight_to_lbs(handicap)
        or_val   = ride.get('official_rating','')

        hg_list = ride.get('headgear') or []
        aid = ' '.join(HEADGEAR_SYMBOLS.get(h.get('symbol',''), h.get('symbol','')) for h in hg_list)

        odds_str = (ride.get('betting') or {}).get('current_odds','-') or '-'

        # CDB from embedded previous_results (course/distance win history)
        prev_results = horse.get('previous_results') or []
        has_c = any(normalise_course(pr.get('course_name','')) == course and pr.get('position')==1
                    for pr in prev_results)
        has_d = any(str(pr.get('distance','')).strip() == str(dist).strip() and pr.get('position')==1
                    for pr in prev_results)
        has_cd = any(normalise_course(pr.get('course_name','')) == course
                     and str(pr.get('distance','')).strip() == str(dist).strip()
                     and pr.get('position')==1 for pr in prev_results)
        cdb = 'CD' if has_cd else ('C D' if (has_c and has_d) else 'C' if has_c else 'D' if has_d else '')

        rows.append({
            'Id': len(rows)+1, 'RaceID': job['race_id'],
            'Course': course, 'Time': race_time, 'Dist': str(dist),
            'RaceDesc': str(title), 'Class': cls, 'AgeLimit': '', 'Value': '',
            'Going': going, 'Run': ran, 'No': len(rows)+1,
            'Draw': draw, 'Form': form, 'Horse': horse_name, 'Days': days,
            'CDB': cdb, 'Age': age, 'Weight': handicap, 'Aid': aid,
            'Trainer': trainer_name, 'Jockey': jockey_name, 'Allow': '',
            'OR': or_val, 'AdjOR': '',
            'Odds': odds_str, 'FcstOdds': odds_str, 'WtLbs': wt_lbs or '',
            'Sire': '', 'Dam': '', 'DamSire': '', 'Sex': '',
            'IsJump': is_jump, 'Type': type_code,
            'MarketId': '', 'SelectionId': '',
        })

    if (i+1) % 10 == 0:
        print(f"  ...{i+1}/{len(race_jobs)} races fetched ({len(rows)} runners so far)")

if not rows:
    print("No runners found for this date.")
    os.makedirs('docs', exist_ok=True)
    with open('docs/index.html','w') as f:
        f.write(f'<title>No Races</title><p>No GB/IE runners found for {date_str}.</p>')
    sys.exit(0)

df = pd.DataFrame(rows)
df = df.sort_values(['Time','Course','No']).reset_index(drop=True)
df['Id'] = df.index + 1
df.to_csv(out_file, index=False)

races = df.groupby(['Time','Course']).ngroups
flat  = (~df['IsJump']).sum()
jumps = df['IsJump'].sum()
no_odds = (df['Odds'] == '-').sum()

print(f"\nDone: {out_file}")
print(f"  {len(df):,} runners across {races} races  ({flat} flat / {jumps} jumps)")
if no_odds: print(f"  NOTE: {no_odds} runners have no current odds")
print(f"\nNext: python predict_actions.py {out_file}")
