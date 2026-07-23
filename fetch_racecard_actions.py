"""
fetch_racecard_actions.py  –  GitHub Actions version of fetch_racecard.py.

Differences from fetch_racecard.py:
  - Credentials read from environment variables (no hardcoded values)
  - Uses client.login() (non-interactive cert login) instead of login_interactive()
  - CDB flag omitted (races_master not in repo); predict_actions.py handles gracefully
  - No --going override; always scrapes SportingLife

Usage (local test):
  set BETFAIR_USERNAME=jackrobertson50@hotmail.com
  set BETFAIR_PASSWORD=Tynie1874!
  set BETFAIR_APP_KEY=zMcr5PWYHKhFzQIr
  python fetch_racecard_actions.py

GitHub Actions sets the above env vars from repository secrets automatically.
"""

import betfairlightweight
from betfairlightweight import filters
import pandas as pd
import numpy as np
import requests, re, sys, os, io, json
from datetime import datetime, timezone, date, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Credentials from environment ───────────────────────────────────────────────
BETFAIR_USERNAME = os.environ['BETFAIR_USERNAME']
BETFAIR_PASSWORD = os.environ['BETFAIR_PASSWORD']
BETFAIR_APP_KEY  = os.environ['BETFAIR_APP_KEY']

target_date = date.today()
date_str    = target_date.strftime('%Y-%m-%d')
date_label  = target_date.strftime('%y%m%d')
out_file    = f"{date_label}racecard.csv"
countries   = ['GB', 'IE']

print(f"Fetching racecard for {date_str}  ->  {out_file}")

# ── Going scraper (SportingLife __NEXT_DATA__) ────────────────────────────────
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

def normalise_course(name):
    n = re.sub(r'\s*\(.*?\)', '', str(name)).strip().upper()
    n = re.sub(r'[^A-Z0-9 ]', '', n).strip()
    return COURSE_ALIASES.get(n[:4], n)

def fetch_going(target_date):
    going_map = {}
    date_fmt  = target_date.strftime('%Y-%m-%d')
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/124.0.0.0 Safari/537.36'),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-GB,en;q=0.9',
    }
    try:
        url  = f'https://www.sportinglife.com/racing/racecards/{date_fmt}'
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            nd = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                           resp.text, re.S)
            if nd:
                data     = json.loads(nd.group(1))
                meetings = (data.get('props', {})
                               .get('pageProps', {})
                               .get('meetings', []))
                for meeting in meetings:
                    summary = meeting.get('meeting_summary', {})
                    course_field = summary.get('course', '')
                    if isinstance(course_field, dict):
                        course = course_field.get('name', '')
                    else:
                        course = str(course_field)
                    going = summary.get('going', '')
                    if course and going:
                        going_map[normalise_course(course)] = going
        print(f"  [Going] {len(going_map)} courses: {list(going_map.items())[:4]}")
    except Exception as e:
        print(f"  [Going] SportingLife failed: {e}")
    return going_map

def parse_market_name(market_name):
    dist_m = re.search(r'(\d+m(?:\s*\d+f)?|\d+f)', market_name, re.I)
    dist   = dist_m.group(1).replace(' ', '') if dist_m else ''
    cls_m  = re.search(r'\bCls?\s*(\d)\b', market_name, re.I)
    cls    = int(cls_m.group(1)) if cls_m else None
    return dist, cls

def parse_race_type(race_type_str):
    rt = str(race_type_str or '').lower()
    if 'hurdle' in rt or 'nhf' in rt or 'bumper' in rt:
        return True, 'h' if 'hurdle' in rt else 'b'
    if 'chase' in rt or 'steeplechase' in rt:
        return True, 'c'
    return False, 'f'

def forecast_odds(num, den):
    try:
        n, d = int(num), int(den)
        if n <= 0 or d <= 0: return '-'
        return f"{n}/{d}"
    except:
        return '-'

# ── Connect to Betfair (standard username/password login) ────────────────────
# login_interactive() uses the standard Betfair endpoint (not the bot endpoint)
# so no cert registration with Betfair is needed.
# We still pass certs for TLS if they exist, but they don't need to be registered.
certs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'certs')
client_kwargs = dict(
    username=BETFAIR_USERNAME,
    password=BETFAIR_PASSWORD,
    app_key=BETFAIR_APP_KEY,
)
if os.path.isdir(certs_path):
    client_kwargs['certs'] = certs_path
client = betfairlightweight.APIClient(**client_kwargs)
client.login_interactive()
print("Connected to Betfair.\n")

# ── Fetch going ────────────────────────────────────────────────────────────────
going_lookup  = fetch_going(target_date)
going_default = 'Unknown'

# ── Fetch markets ──────────────────────────────────────────────────────────────
start_of_day = f"{date_str}T00:00:00Z"
end_of_day   = f"{date_str}T23:59:59Z"

mf = filters.market_filter(
    event_type_ids=['7'],
    market_countries=countries,
    market_type_codes=['WIN'],
    market_start_time={'from': start_of_day, 'to': end_of_day},
)

print(f"Fetching WIN markets for {', '.join(countries)} on {date_str}...")
markets = client.betting.list_market_catalogue(
    filter=mf,
    market_projection=[
        'MARKET_START_TIME', 'RUNNER_DESCRIPTION', 'RUNNER_METADATA',
        'EVENT', 'MARKET_DESCRIPTION', 'EVENT_TYPE',
    ],
    max_results=500,
    sort='FIRST_TO_START',
)
print(f"  Found {len(markets)} WIN markets")

# ── Get current prices ────────────────────────────────────────────────────────
market_ids = []
for m in markets:
    mid = m.market_id if hasattr(m, 'market_id') else m.get('marketId')
    if mid: market_ids.append(mid)

price_map = {}
print(f"Fetching prices for {len(market_ids)} markets...")
for i in range(0, len(market_ids), 40):
    chunk = market_ids[i:i+40]
    try:
        books = client.betting.list_market_book(
            market_ids=chunk,
            price_projection=filters.price_projection(
                price_data=['EX_BEST_OFFERS'],
                ex_best_offers_overrides=filters.ex_best_offers_overrides(
                    best_prices_depth=1, rollup_model='STAKE', rollup_limit=5)
            ),
            order_projection='EXECUTABLE',
            match_projection='NO_ROLLUP',
        )
        for book in books:
            mid = book.market_id if hasattr(book,'market_id') else book.get('marketId')
            runners_data = book.runners if hasattr(book,'runners') else book.get('runners',[])
            price_map[mid] = {}
            for r in runners_data:
                sel_id = r.selection_id if hasattr(r,'selection_id') else r.get('selectionId')
                ex     = r.ex if hasattr(r,'ex') else r.get('ex',{})
                avail  = (ex.available_to_back if hasattr(ex,'available_to_back')
                          else (ex or {}).get('availableToBack', []))
                last_t = (r.last_price_traded if hasattr(r,'last_price_traded')
                          else r.get('lastPriceTraded'))
                if avail:
                    p = avail[0].price if hasattr(avail[0],'price') else avail[0].get('price')
                elif last_t:
                    p = last_t
                else:
                    p = None
                if p and p > 1:
                    num = round(p - 1, 2)
                    for denom in [1,2,4,5,8,10,20]:
                        candidate = num * denom
                        if abs(candidate - round(candidate)) < 0.05:
                            price_map[mid][sel_id] = f"{int(round(candidate))}/{denom}"
                            break
                    else:
                        price_map[mid][sel_id] = f"{num:.1f}/1"
    except Exception as e:
        print(f"  Price fetch error chunk {i}: {e}")

# ── Build racecard rows ────────────────────────────────────────────────────────
rows = []
race_id_counter  = 0
current_race_key = None

for m in markets:
    mid       = m.market_id   if hasattr(m,'market_id')   else m.get('marketId')
    mname     = m.market_name if hasattr(m,'market_name') else m.get('marketName','')
    mtime_obj = (m.market_start_time if hasattr(m,'market_start_time')
                 else m.get('marketStartTime'))
    event     = m.event if hasattr(m,'event') else m.get('event',{})
    desc      = m.description if hasattr(m,'description') else m.get('description')
    runners   = m.runners if hasattr(m,'runners') else m.get('runners',[])

    ename = (event.name if hasattr(event,'name') else
             (event.get('name','') if isinstance(event,dict) else ''))
    course_raw = re.sub(r'\s+\d+.*$', '', ename).strip()
    course     = normalise_course(course_raw)

    if hasattr(mtime_obj, 'strftime'):
        race_time = (mtime_obj + timedelta(hours=1)).strftime('%H:%M')
    elif mtime_obj:
        t = datetime.fromisoformat(str(mtime_obj).replace('Z','+00:00'))
        race_time = (t + timedelta(hours=1)).strftime('%H:%M')
    else:
        race_time = '??:??'

    dist, cls = parse_market_name(mname)

    race_type_str = ''
    if desc:
        race_type_str = (desc.race_type if hasattr(desc,'race_type')
                         else (desc.get('raceType','') if isinstance(desc,dict) else ''))
    is_jump, type_code = parse_race_type(race_type_str)
    going = going_lookup.get(course, going_default)
    ran   = len(runners)

    race_key = f"{race_time}_{course}"
    if race_key != current_race_key:
        race_id_counter += 1
        current_race_key = race_key

    mprices = price_map.get(mid, {})

    for runner_num, r in enumerate(runners, 1):
        rname  = (r.runner_name if hasattr(r,'runner_name') else
                  r.get('runnerName', r.get('name','?')))
        sel_id = (r.selection_id if hasattr(r,'selection_id') else r.get('selectionId'))
        meta   = (r.metadata if hasattr(r,'metadata') else r.get('metadata', {})) or {}

        or_val  = int(meta.get('OFFICIAL_RATING') or 0)
        adj_or  = int(meta.get('ADJUSTED_RATING') or 0)
        if or_val  >= 999: or_val  = 0
        if adj_or  >= 999: adj_or  = 0
        or_final = or_val if or_val > 0 else adj_or

        draw       = meta.get('STALL_DRAW')
        trainer    = meta.get('TRAINER_NAME', '')
        jockey     = meta.get('JOCKEY_NAME', '')
        form       = meta.get('FORM', '')
        age        = meta.get('AGE')
        weight_lbs = meta.get('WEIGHT_VALUE')
        days       = meta.get('DAYS_SINCE_LAST_RUN')
        wearing    = meta.get('WEARING', '')
        jock_claim = int(meta.get('JOCKEY_CLAIM') or 0)
        fcst_num   = meta.get('FORECASTPRICE_NUMERATOR')
        fcst_den   = meta.get('FORECASTPRICE_DENOMINATOR')
        sire       = meta.get('SIRE_NAME', '')
        dam        = meta.get('DAM_NAME', '')
        damsire    = meta.get('DAMSIRE_NAME', '')
        sex        = meta.get('SEX_TYPE', '')

        fcst_str  = forecast_odds(fcst_num, fcst_den) if fcst_num else '-'
        exch_odds = mprices.get(sel_id, '-')
        allow     = f"{jock_claim}" if jock_claim and jock_claim > 0 else ''

        if weight_lbs:
            st  = int(float(weight_lbs)) // 14
            lbs = int(float(weight_lbs)) % 14
            weight_str = f"{st}-{lbs}"
        else:
            weight_str = ''

        rows.append({
            'Id':        len(rows) + 1,
            'RaceID':    race_id_counter,
            'Course':    course,
            'Time':      race_time,
            'Dist':      dist,
            'RaceDesc':  mname,
            'Class':     cls or '',
            'AgeLimit':  '',
            'Value':     '',
            'Going':     going,
            'Run':       ran,
            'No':        runner_num,
            'Draw':      draw or '',
            'Form':      form,
            'Horse':     rname,
            'Days':      days or '',
            'CDB':       '',          # not available without races_master
            'Age':       age or '',
            'Weight':    weight_str,
            'Aid':       wearing or '',
            'Trainer':   trainer,
            'Jockey':    jockey,
            'Allow':     allow,
            'OR':        or_final if or_final > 0 else '',
            'AdjOR':     adj_or if adj_or > 0 else '',
            'Odds':      exch_odds,
            'FcstOdds':  fcst_str,
            'WtLbs':     int(float(weight_lbs)) if weight_lbs else '',
            'Sire':      sire,
            'Dam':       dam,
            'DamSire':   damsire,
            'Sex':       sex,
            'IsJump':    is_jump,
            'Type':      type_code,
            'MarketId':  mid,
            'SelectionId': sel_id,
        })

# ── Output ─────────────────────────────────────────────────────────────────────
if not rows:
    print("No runners found — writing empty picks page.")
    os.makedirs('docs', exist_ok=True)
    with open('docs/index.html', 'w') as f:
        f.write(f'<title>No Races</title><p>No GB/IE races found for {date_str}.</p>')
    sys.exit(0)

df = pd.DataFrame(rows)
df = df.sort_values(['Time','Course','No']).reset_index(drop=True)
df['Id'] = df.index + 1
df.to_csv(out_file, index=False)

races = df.groupby(['Time','Course']).ngroups
flat  = (~df['IsJump']).sum()
jumps = df['IsJump'].sum()
no_go = (df['Going'] == 'Unknown').sum()

print(f"\nDone: {out_file}")
print(f"  {len(df):,} runners across {races} races  ({flat} flat / {jumps} jumps)")
if no_go: print(f"  NOTE: {no_go} runners have Going=Unknown")
print(f"\nNext step: python predict_actions.py {out_file}")
