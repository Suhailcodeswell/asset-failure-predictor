"""
Build trip-level feature CSV v3: one row per rail manifest with all features
computed BEFORE the manifest date (no data leakage).

v3 additions over v2:
- Telematics event history (shutdowns, alarms, low fuel, restarts before trip)
- Battery voltage & fuel % from telematics
- Engine hours from meter reading history
- Alarm severity scoring (red/yellow/green from reference)
- Container-level repairs (separate from reefer repairs)
- Fuel fill history (Comdata)
- Major PM types (PM-012, PM-013, PM-014, PM-070, PM-071)
- Ambient weather: origin/dest/waypoint temps, transit temp swing (Open-Meteo)
- Operating mode (Continuous vs Cycle Sentry from telematics)
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import re
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

print('=' * 80)
print('BUILDING TRIP FEATURES v3')
print('=' * 80)

# ======================================================================
# LOAD DATA SOURCES
# ======================================================================

# === REPAIRS (reefer unit) ===
print('\nLoading reefer repairs...')
repairs = pd.read_excel('data/raw/repairs/Updated Reefer Unit Repairs 2024-2025.xlsx')
repairs['COMPCODE'] = repairs['COMPCODE'].str.strip()
repairs['REPREASON'] = repairs['REPREASON'].str.strip()
repairs['UNITNUMBER'] = repairs['UNITNUMBER'].str.strip()
repairs['DESCRIP'] = repairs['DESCRIP'].str.strip()
repairs['SHOPID'] = repairs['SHOPID'].str.strip()
repairs['OPENED'] = pd.to_datetime(repairs['OPENED'])
repairs['SumOfLINETOTAL'] = pd.to_numeric(repairs['SumOfLINETOTAL'], errors='coerce').fillna(0)
print(f'  Reefer repairs: {len(repairs)} rows')

# === FLEET ===
print('Loading fleet...')
fleet = pd.read_csv('data/processed/training/active_intmdl_fleet.csv')
fleet['container_num'] = fleet['container_num'].astype(str).str.strip()
fleet['reefer_unit'] = fleet['reefer_unit'].str.strip()
trailer_to_reefer = dict(zip(fleet['container_num'], fleet['reefer_unit']))
reefer_to_slxi = dict(zip(fleet['reefer_unit'], fleet['is_slxi']))
reefer_to_container = dict(zip(fleet['reefer_unit'], fleet['container_num']))

# Fleet characteristics
fleet_info = {}
for _, row in fleet.iterrows():
    fleet_info[row['reefer_unit']] = {
        'is_slxi': row['is_slxi'],
        'is_thermoking': row.get('is_thermoking', 0),
        'is_carrier': row.get('is_carrier', 0),
        'reefer_model': row.get('reefer_model', ''),
        'reefer_year': row.get('reefer_year', 0),
        'container_year': row.get('container_year', 0),
        'install_date': pd.to_datetime(row.get('install_date', None), errors='coerce'),
        'in_service_date': pd.to_datetime(row.get('in_service_date', None), errors='coerce'),
    }

# Build container_num -> reefer_unit mapping for telematics/container sources
# Container IDs in telematics look like "TNXU535027" — extract numeric portion
def extract_container_num(container_id):
    """Extract numeric container number from full container ID."""
    if pd.isna(container_id):
        return None
    s = str(container_id).strip()
    # Remove common prefixes (TNXU, HRTU, TXCU, CSNU, CNRU, etc.)
    nums = re.sub(r'[^0-9]', '', s)
    return nums if nums else None

# === TELEMATICS — shutdown labels (same as v2) ===
print('Loading telematics (Combined)...')
tele = pd.read_excel('data/raw/telematics/Combined_Telematics_2025.xlsx')
tele['CONTAINER'] = tele['CONTAINER'].astype(str).str.strip()
tele['SHUTDOWN_DATE_TIME'] = pd.to_datetime(tele['SHUTDOWN_DATE_TIME'], errors='coerce')

# Manifest corrections
mc = pd.read_csv('data/raw/telematics/manifest_corrections.csv')
mc_shut = mc[mc['EVENT_TYPE'] == 'SHUTDOWN'].copy()
mc_shut['CONTAINER'] = mc_shut['CONTAINER'].astype(str).str.strip()
mc_shut['SHUTDOWN_DATE_TIME'] = pd.to_datetime(mc_shut['SHUTDOWN_DATE_TIME'], errors='coerce')

correction_lookup = {}
for _, row in mc_shut.iterrows():
    if pd.notna(row['VERIFIED_MANIFEST']):
        key = (row['CONTAINER'], str(row['SHUTDOWN_DATE_TIME']))
        correction_lookup[key] = str(int(row['VERIFIED_MANIFEST']))

shutdown_manifests = set()
for _, row in tele.iterrows():
    container = row['CONTAINER']
    sdt = str(row['SHUTDOWN_DATE_TIME'])
    key = (container, sdt)
    if key in correction_lookup:
        manifest = correction_lookup[key]
    else:
        manifest = str(int(row['MANIFEST'])) if pd.notna(row.get('MANIFEST')) else None
    if manifest and manifest != 'nan':
        shutdown_manifests.add(manifest)

print(f'  Verified shutdown manifests: {len(shutdown_manifests)}')

# === QUARTERLY TELEMATICS (all event types) ===
print('Loading quarterly telematics (alarms, low fuel, restarts, not reporting)...')
quarterly_files = [
    'data/raw/telematics/quarterly/Daily Telematics Update -YTD 2025 Q1.xlsx',
    'data/raw/telematics/quarterly/Daily Telematics Update -YTD 2025 Q2.xlsx',
    'data/raw/telematics/quarterly/Daily Telematics Update -YTD 2025 Q3.xlsx',
    'data/raw/telematics/quarterly/Daily Telematics Update -YTD 2025 Q4.xlsx',
]

telem_events = []  # list of (container_num, event_date, event_type, alarm_code, battery_voltage, fuel_pct, mode)


def _find_sheet(sheet_names, target):
    """Case-insensitive sheet name lookup."""
    for s in sheet_names:
        if s.upper() == target.upper():
            return s
    return None


def _read_telem_sheet(xf, target, header=1):
    """Read a telematics sheet with case-insensitive name and auto-header detection.
    Some quarterly files (Q4) have an extra title row, shifting the real header to row 2."""
    sname = _find_sheet(xf.sheet_names, target)
    if sname is None:
        return None
    df = pd.read_excel(xf, sheet_name=sname, header=header)
    # Detect wrong header: if no column contains 'CONTAINER', the header row is off
    if not any('CONTAINER' in str(c).upper() for c in df.columns):
        df = pd.read_excel(xf, sheet_name=sname, header=header + 1)
        if not any('CONTAINER' in str(c).upper() for c in df.columns):
            return None  # cannot find usable header
    return df


for qf in quarterly_files:
    if not os.path.exists(qf):
        print(f'  Skipping {qf} (not found)')
        continue
    xf = pd.ExcelFile(qf)

    # SHUTDOWN sheet
    df_s = _read_telem_sheet(xf, 'SHUTDOWN')
    if df_s is not None:
        date_col = 'SHUTDOWN DATE/TIME' if 'SHUTDOWN DATE/TIME' in df_s.columns else 'Date'
        for _, r in df_s.iterrows():
            cnum = extract_container_num(r.get('CONTAINER'))
            dt = pd.to_datetime(r.get(date_col), errors='coerce')
            if cnum and pd.notna(dt):
                bv = pd.to_numeric(r.get('Battery Voltage', r.get('BATTERY_VOLTAGE')), errors='coerce')
                fuel = pd.to_numeric(r.get('Fuel', r.get('FUEL')), errors='coerce')
                mode = str(r.get('MODE', ''))
                alarm = str(r.get('ALARM CODE', r.get('ALARM_CODE', '')))
                telem_events.append((cnum, dt, 'SHUTDOWN', alarm, bv, fuel, mode))

    # ALARMS sheet
    df_a = _read_telem_sheet(xf, 'ALARMS')
    if df_a is not None:
        for _, r in df_a.iterrows():
            cnum = extract_container_num(r.get('CONTAINER'))
            dt = pd.to_datetime(r.get('Date'), errors='coerce')
            if cnum and pd.notna(dt):
                bv = pd.to_numeric(r.get('Battery Voltage'), errors='coerce')
                fuel = pd.to_numeric(r.get('Fuel'), errors='coerce')
                mode = str(r.get('MODE', ''))
                alarm = str(r.get('ALARM CODE', ''))
                telem_events.append((cnum, dt, 'ALARM', alarm, bv, fuel, mode))

    # LOW FUEL sheet
    df_lf = _read_telem_sheet(xf, 'LOW FUEL')
    if df_lf is not None:
        for _, r in df_lf.iterrows():
            cnum = extract_container_num(r.get('CONTAINER'))
            dt = pd.to_datetime(r.get('Date'), errors='coerce')
            if cnum and pd.notna(dt):
                bv = pd.to_numeric(r.get('Battery Voltage'), errors='coerce')
                fuel = pd.to_numeric(r.get('Fuel', r.get('Engine Hours')), errors='coerce')
                mode = str(r.get('MODE', ''))
                telem_events.append((cnum, dt, 'LOW_FUEL', '96', bv, fuel, mode))

    # RESTARTED sheet
    df_r = _read_telem_sheet(xf, 'RESTARTED')
    if df_r is not None:
        date_col = 'RESTART DATE/TIME' if 'RESTART DATE/TIME' in df_r.columns else 'Date'
        for _, r in df_r.iterrows():
            cnum = extract_container_num(r.get('CONTAINER'))
            dt = pd.to_datetime(r.get(date_col), errors='coerce')
            if cnum and pd.notna(dt):
                bv = pd.to_numeric(r.get('Battery Voltage'), errors='coerce')
                fuel = pd.to_numeric(r.get('FUEL', r.get('Fuel')), errors='coerce')
                mode = str(r.get('MODE', ''))
                telem_events.append((cnum, dt, 'RESTARTED', '', bv, fuel, mode))

    # NOT REPORTING sheet
    df_nr = _read_telem_sheet(xf, 'NOT REPORTING')
    if df_nr is not None:
        for _, r in df_nr.iterrows():
            cnum = extract_container_num(r.get('CONTAINER'))
            dt = pd.to_datetime(r.get('Date'), errors='coerce')
            if cnum and pd.notna(dt):
                telem_events.append((cnum, dt, 'NOT_REPORTING', '', np.nan, np.nan, ''))

print(f'  Total telematics events loaded: {len(telem_events)}')

# Build per-reefer-unit sorted event lists
telem_by_unit = defaultdict(list)
for cnum, dt, etype, alarm, bv, fuel, mode in telem_events:
    reefer = trailer_to_reefer.get(cnum)
    if reefer:
        telem_by_unit[reefer].append({
            'date': dt, 'type': etype, 'alarm': alarm,
            'battery': bv, 'fuel': fuel, 'mode': mode,
        })

for unit in telem_by_unit:
    telem_by_unit[unit].sort(key=lambda x: x['date'])

print(f'  Units with telematics history: {len(telem_by_unit)}')

# === ALARM SEVERITY REFERENCE ===
print('Loading alarm severity reference...')
alarm_sev = pd.read_csv('data/processed/reference/alarm_severity_classification.csv')
alarm_severity_map = {}
for _, r in alarm_sev.iterrows():
    code = str(r['Code']).strip()
    sev = str(r.get('Official_Severity', '')).strip()
    alarm_severity_map[code] = sev
print(f'  Alarm codes mapped: {len(alarm_severity_map)}')

# === ENGINE HOURS (METER READINGS) ===
print('Loading meter reading history...')
meter = pd.read_excel('data/raw/hours meter reading/Meter Reading History.xlsx')
meter['UNITNUMBER'] = meter['UNITNUMBER'].astype(str).str.strip()
meter['READDAY'] = pd.to_datetime(meter['READDAY'], errors='coerce')
meter['READING'] = pd.to_numeric(meter['READING'], errors='coerce')
# Filter: valid dates (before 2030), valid readings, hour meter only
meter = meter[
    (meter['METERTYPE'] == 'HOUR METER') &
    (meter['READDAY'] < '2030-01-01') &
    (meter['READING'] > 0)
].copy()
meter = meter.sort_values(['UNITNUMBER', 'READDAY'])

# Build per-unit sorted reading list
meter_by_unit = {}
for unit, grp in meter.groupby('UNITNUMBER'):
    readings = grp[['READDAY', 'READING']].drop_duplicates().sort_values('READDAY')
    meter_by_unit[unit] = list(zip(readings['READDAY'], readings['READING']))

print(f'  Meter readings: {len(meter)} rows, {len(meter_by_unit)} units')

# === CONTAINER REPAIRS ===
print('Loading container repairs...')
crep = pd.read_excel('data/raw/repairs/Container Repair Detail.xlsx')
crep['UNITNUMBER'] = crep['UNITNUMBER'].astype(str).str.strip()
crep['OPENED'] = pd.to_datetime(crep['OPENED'], errors='coerce')
crep['REPREASON'] = crep['REPREASON'].str.strip()
crep['COMPCODE'] = crep['COMPCODE'].str.strip()
crep['SumOfLINETOTAL'] = pd.to_numeric(crep['SumOfLINETOTAL'], errors='coerce').fillna(0)

# Aggregate by order
crep_orders = crep.groupby(['UNITNUMBER', 'ORDERNUM']).agg(
    date=('OPENED', 'min'),
    total_cost=('SumOfLINETOTAL', 'sum'),
    repreason=('REPREASON', 'first'),
).reset_index()
crep_orders['date'] = pd.to_datetime(crep_orders['date'])

# Build per-container sorted repair list (keyed by container_num string)
container_repairs_by_num = {}
for cnum, grp in crep_orders.groupby('UNITNUMBER'):
    container_repairs_by_num[cnum] = grp.sort_values('date').to_dict('records')

print(f'  Container repairs: {len(crep_orders)} orders, {len(container_repairs_by_num)} containers')

# === FUEL DATA ===
print('Loading fuel data...')
fuel_frames = []
for fpath in ['data/raw/fuel data/MMA 2024 Fuel Data.xlsx', 'data/raw/fuel data/MMA 2025 Fuel Data.xlsx']:
    if os.path.exists(fpath):
        f = pd.read_excel(fpath)
        fuel_frames.append(f)
fuel_all = pd.concat(fuel_frames, ignore_index=True)
fuel_all['Fuel Date Time'] = pd.to_datetime(fuel_all['Fuel Date Time'], errors='coerce')
fuel_all['Reefer Liters'] = pd.to_numeric(fuel_all['Reefer Liters'], errors='coerce')

# Clean trailer numbers
def clean_trailer(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    nums = re.sub(r'[^0-9]', '', s)
    # Only keep reasonable container numbers (5-6 digits)
    if nums and 5 <= len(nums) <= 6:
        return nums
    return None

fuel_all['container_num'] = fuel_all['Assigned Trailer'].apply(clean_trailer)
fuel_valid = fuel_all.dropna(subset=['container_num', 'Fuel Date Time']).copy()
fuel_valid = fuel_valid[fuel_valid['Reefer Liters'] > 0]

# Build per-container sorted fuel list
fuel_by_container = {}
for cnum, grp in fuel_valid.groupby('container_num'):
    entries = grp[['Fuel Date Time', 'Reefer Liters']].sort_values('Fuel Date Time')
    fuel_by_container[cnum] = list(zip(entries['Fuel Date Time'], entries['Reefer Liters']))

print(f'  Fuel records: {len(fuel_valid)} valid, {len(fuel_by_container)} containers')

# === WEATHER DATA (Open-Meteo historical) ===
print('Loading weather data from Open-Meteo...')

HUB_COORDS = {
    'CN TOR': (43.65, -79.38),
    'CN CGY': (51.05, -114.07),
    'CN EDM': (53.55, -113.49),
    'CN VCR': (49.25, -123.10),
    'CN MTL': (45.50, -73.57),
    'CN WPG': (49.90, -97.14),
    'CN MOC': (46.09, -64.77),
    'CN REG': (50.45, -104.62),
    'CN SAS': (52.13, -106.67),
    'CN HAL': (44.65, -63.57),
}

# Route waypoints — intermediate hubs the train passes through
ROUTE_WAYPOINTS = {
    ('CN TOR', 'CN VCR'): ['CN TOR', 'CN WPG', 'CN SAS', 'CN EDM', 'CN CGY', 'CN VCR'],
    ('CN VCR', 'CN TOR'): ['CN VCR', 'CN CGY', 'CN EDM', 'CN SAS', 'CN WPG', 'CN TOR'],
    ('CN TOR', 'CN CGY'): ['CN TOR', 'CN WPG', 'CN SAS', 'CN REG', 'CN CGY'],
    ('CN CGY', 'CN TOR'): ['CN CGY', 'CN REG', 'CN SAS', 'CN WPG', 'CN TOR'],
    ('CN TOR', 'CN EDM'): ['CN TOR', 'CN WPG', 'CN SAS', 'CN EDM'],
    ('CN EDM', 'CN TOR'): ['CN EDM', 'CN SAS', 'CN WPG', 'CN TOR'],
    ('CN TOR', 'CN WPG'): ['CN TOR', 'CN WPG'],
    ('CN WPG', 'CN TOR'): ['CN WPG', 'CN TOR'],
    ('CN MTL', 'CN VCR'): ['CN MTL', 'CN TOR', 'CN WPG', 'CN SAS', 'CN EDM', 'CN CGY', 'CN VCR'],
    ('CN VCR', 'CN MTL'): ['CN VCR', 'CN CGY', 'CN EDM', 'CN SAS', 'CN WPG', 'CN TOR', 'CN MTL'],
    ('CN MTL', 'CN CGY'): ['CN MTL', 'CN TOR', 'CN WPG', 'CN SAS', 'CN REG', 'CN CGY'],
    ('CN CGY', 'CN MTL'): ['CN CGY', 'CN REG', 'CN SAS', 'CN WPG', 'CN TOR', 'CN MTL'],
    ('CN MTL', 'CN EDM'): ['CN MTL', 'CN TOR', 'CN WPG', 'CN SAS', 'CN EDM'],
    ('CN EDM', 'CN MTL'): ['CN EDM', 'CN SAS', 'CN WPG', 'CN TOR', 'CN MTL'],
    ('CN MTL', 'CN WPG'): ['CN MTL', 'CN TOR', 'CN WPG'],
    ('CN WPG', 'CN MTL'): ['CN WPG', 'CN TOR', 'CN MTL'],
    ('CN TOR', 'CN MTL'): ['CN TOR', 'CN MTL'],
    ('CN MTL', 'CN TOR'): ['CN MTL', 'CN TOR'],
    ('CN VCR', 'CN CGY'): ['CN VCR', 'CN CGY'],
    ('CN CGY', 'CN VCR'): ['CN CGY', 'CN VCR'],
    ('CN VCR', 'CN EDM'): ['CN VCR', 'CN CGY', 'CN EDM'],
    ('CN EDM', 'CN VCR'): ['CN EDM', 'CN CGY', 'CN VCR'],
    ('CN CGY', 'CN EDM'): ['CN CGY', 'CN EDM'],
    ('CN EDM', 'CN CGY'): ['CN EDM', 'CN CGY'],
    ('CN CGY', 'CN WPG'): ['CN CGY', 'CN REG', 'CN SAS', 'CN WPG'],
    ('CN WPG', 'CN CGY'): ['CN WPG', 'CN SAS', 'CN REG', 'CN CGY'],
    ('CN EDM', 'CN WPG'): ['CN EDM', 'CN SAS', 'CN WPG'],
    ('CN WPG', 'CN EDM'): ['CN WPG', 'CN SAS', 'CN EDM'],
    ('CN MOC', 'CN TOR'): ['CN MOC', 'CN MTL', 'CN TOR'],
    ('CN TOR', 'CN MOC'): ['CN TOR', 'CN MTL', 'CN MOC'],
    ('CN MOC', 'CN MTL'): ['CN MOC', 'CN MTL'],
    ('CN MTL', 'CN MOC'): ['CN MTL', 'CN MOC'],
    ('CN HAL', 'CN TOR'): ['CN HAL', 'CN MOC', 'CN MTL', 'CN TOR'],
    ('CN TOR', 'CN HAL'): ['CN TOR', 'CN MTL', 'CN MOC', 'CN HAL'],
    ('CN HAL', 'CN MTL'): ['CN HAL', 'CN MOC', 'CN MTL'],
    ('CN MTL', 'CN HAL'): ['CN MTL', 'CN MOC', 'CN HAL'],
}

# Fetch weather for all hubs for 2024-2025 (covers all trips)
weather_cache_path = 'data/processed/training/weather_cache.csv'
if os.path.exists(weather_cache_path):
    print('  Loading cached weather data...')
    weather_df = pd.read_csv(weather_cache_path, parse_dates=['date'])
    weather_lookup = {}
    for _, r in weather_df.iterrows():
        weather_lookup[(r['hub'], r['date'].strftime('%Y-%m-%d'))] = {
            'temp_mean': r['temp_mean_f'],
            'temp_min': r['temp_min_f'],
            'temp_max': r['temp_max_f'],
        }
    print(f'  Loaded {len(weather_lookup)} cached weather records')
else:
    print('  Fetching weather from Open-Meteo API (this may take a minute)...')
    import urllib.request
    import json

    weather_lookup = {}
    weather_rows = []

    # Determine weather years dynamically: previous year + current year
    _current_year = pd.Timestamp.now().year
    _weather_years = [_current_year - 1, _current_year]
    for hub, (lat, lon) in HUB_COORDS.items():
        for year in _weather_years:
            url = (
                f'https://archive-api.open-meteo.com/v1/archive?'
                f'latitude={lat}&longitude={lon}'
                f'&start_date={year}-01-01&end_date={year}-12-31'
                f'&daily=temperature_2m_mean,temperature_2m_min,temperature_2m_max'
                f'&temperature_unit=fahrenheit&timezone=America/Toronto'
            )
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                daily = data.get('daily', {})
                dates = daily.get('time', [])
                t_mean = daily.get('temperature_2m_mean', [])
                t_min = daily.get('temperature_2m_min', [])
                t_max = daily.get('temperature_2m_max', [])
                for i, d in enumerate(dates):
                    weather_lookup[(hub, d)] = {
                        'temp_mean': t_mean[i] if i < len(t_mean) else None,
                        'temp_min': t_min[i] if i < len(t_min) else None,
                        'temp_max': t_max[i] if i < len(t_max) else None,
                    }
                    weather_rows.append({
                        'hub': hub, 'date': d,
                        'temp_mean_f': t_mean[i] if i < len(t_mean) else None,
                        'temp_min_f': t_min[i] if i < len(t_min) else None,
                        'temp_max_f': t_max[i] if i < len(t_max) else None,
                    })
                print(f'    {hub} {year}: {len(dates)} days')
            except Exception as e:
                print(f'    WARNING: Failed to fetch {hub} {year}: {e}')
            time.sleep(0.3)  # rate limit courtesy

    # Cache for future runs
    pd.DataFrame(weather_rows).to_csv(weather_cache_path, index=False)
    print(f'  Weather cached to {weather_cache_path} ({len(weather_lookup)} records)')


def get_weather(hub, date):
    """Get weather for a hub on a specific date."""
    key = (hub, date.strftime('%Y-%m-%d'))
    return weather_lookup.get(key, {})


def compute_transit_weather(origin, destination, trip_start, trip_end):
    """Compute weather features along the route during transit."""
    waypoints = ROUTE_WAYPOINTS.get((origin, destination))
    if not waypoints:
        # Fallback: just origin and destination
        waypoints = [origin, destination]

    if pd.isna(trip_end) or pd.isna(trip_start):
        # Use origin weather only
        w = get_weather(origin, trip_start)
        return {
            'origin_temp': w.get('temp_mean'),
            'dest_temp': None,
            'transit_temp_swing': None,
            'transit_min_temp': w.get('temp_min'),
            'transit_max_temp': w.get('temp_max'),
            'origin_dest_temp_diff': None,
            'extreme_cold_flag': 1 if (w.get('temp_min') or 99) < 0 else 0,
            'danger_zone_flag': 1 if 0 <= (w.get('temp_mean') or -99) <= 20 else 0,
        }

    # Estimate days per waypoint
    total_days = max((trip_end - trip_start).total_seconds() / 86400, 1)
    n_waypoints = len(waypoints)

    all_temps = []
    origin_temp = None
    dest_temp = None

    for i, wp in enumerate(waypoints):
        # Estimate what day the train passes through this waypoint
        day_offset = int(i * total_days / max(n_waypoints - 1, 1))
        wp_date = trip_start + pd.Timedelta(days=day_offset)
        w = get_weather(wp, wp_date)
        t_mean = w.get('temp_mean')
        t_min = w.get('temp_min')
        t_max = w.get('temp_max')

        if t_mean is not None:
            all_temps.append(t_mean)
        if t_min is not None:
            all_temps.append(t_min)
        if t_max is not None:
            all_temps.append(t_max)

        if i == 0:
            origin_temp = t_mean
        if i == n_waypoints - 1:
            dest_temp = t_mean

    if all_temps:
        swing = max(all_temps) - min(all_temps)
        min_temp = min(all_temps)
        max_temp = max(all_temps)
    else:
        swing = None
        min_temp = None
        max_temp = None

    if origin_temp is not None and dest_temp is not None:
        od_diff = abs(origin_temp - dest_temp)
    else:
        od_diff = None

    return {
        'origin_temp': origin_temp,
        'dest_temp': dest_temp,
        'transit_temp_swing': swing,
        'transit_min_temp': min_temp,
        'transit_max_temp': max_temp,
        'origin_dest_temp_diff': od_diff,
        'extreme_cold_flag': 1 if (min_temp is not None and min_temp < 0) else 0,
        'danger_zone_flag': 1 if (origin_temp is not None and 0 <= origin_temp <= 20) else 0,
    }

# === RAIL TRIPS ===
print('Loading rail data...')
rail = pd.read_excel('data/raw/mileage/Rail Miles 2024 and 2025.xlsx',
                     sheet_name='Rail Miles Data 2024 & 2025', header=4)
rail['Leg_Trailer1'] = rail['Leg_Trailer1'].astype(str).str.strip()
rail['Leg_Start_Date1'] = pd.to_datetime(rail['Leg_Start_Date1'], errors='coerce')
rail['Leg_End_Date1'] = pd.to_datetime(rail['Leg_End_Date1'], errors='coerce')
rail = rail[~rail['Leg_Trailer1'].str.contains(',', na=False)]
rail = rail[rail['Leg_Trailer1'].str.match(r'^\d+$', na=False)]
rail['reefer_unit'] = rail['Leg_Trailer1'].map(trailer_to_reefer)
rail = rail.dropna(subset=['reefer_unit', 'Leg_Start_Date1']).copy()
rail = rail.sort_values('Leg_Start_Date1').reset_index(drop=True)

print(f'Rail trips to process: {len(rail)}')

# ======================================================================
# PRECOMPUTE REPAIR LOOKUPS (reefer unit — same as v2)
# ======================================================================
print('Building repair lookups...')

# PM events per unit (PM-011 = reefer 360 check)
pm360 = repairs[(repairs['REPREASON'] == 'PM') & (repairs['COMPCODE'] == '000-011')].copy()
pm360['date_only'] = pm360['OPENED'].dt.date
pm_unique = pm360.drop_duplicates(subset=['UNITNUMBER', 'date_only'])
pm_by_unit = pm_unique.groupby('UNITNUMBER')['OPENED'].apply(lambda x: sorted(x)).to_dict()

# Major PM types lookup (PM-012 major service, PM-013 starter, PM-014 alternator,
# PM-070 battery, PM-071 fuel polishing)
MAJOR_PM_CODES = {
    '000-012': 'major_service',
    '000-013': 'starter_replacement',
    '000-014': 'alternator_replacement',
    '000-070': 'battery_replacement',
    '000-071': 'fuel_polishing',
}
major_pm_events = repairs[
    (repairs['REPREASON'] == 'PM') &
    (repairs['COMPCODE'].isin(MAJOR_PM_CODES.keys()))
].copy()

major_pm_by_unit = defaultdict(lambda: defaultdict(list))
for _, r in major_pm_events.iterrows():
    pm_type = MAJOR_PM_CODES[r['COMPCODE']]
    major_pm_by_unit[r['UNITNUMBER']][pm_type].append(r['OPENED'])

# Sort dates
for unit in major_pm_by_unit:
    for pm_type in major_pm_by_unit[unit]:
        major_pm_by_unit[unit][pm_type].sort()

# All repair orders (aggregated by order number)
repair_orders = repairs.groupby(['UNITNUMBER', 'ORDERNUM']).agg(
    date=('OPENED', 'min'),
    total_cost=('SumOfLINETOTAL', 'sum'),
    repreason=('REPREASON', 'first'),
    compcodes=('COMPCODE', lambda x: list(set(x))),
    shopid=('SHOPID', lambda x: x.iloc[0].strip() if len(x) > 0 else ''),
).reset_index()
repair_orders['date'] = pd.to_datetime(repair_orders['date'])

# Per-unit sorted repair list
repairs_by_unit = {}
for unit, grp in repair_orders.groupby('UNITNUMBER'):
    repairs_by_unit[unit] = grp.sort_values('date').to_dict('records')

# Component code categories
COMP_CATS = {
    'compressor': ['082-002'],
    'electrical_charging': ['082-041'],
    'fuel_system': ['082-044'],
    'condenser': ['082-008'],
    'engine': ['082-045'],
    'evaporator': ['082-001'],
    'refrigerant_lines': ['082-031'],
    'coolant': ['082-042'],
    'sheet_metal': ['082-029'],
    'throttle': ['082-036'],
    'starter_cranking': ['082-010'],
    'drier': ['082-012'],
    'diagnosis': ['082-DIA', 'DOR-DIA'],
    'chute_bulkhead': ['082-046'],
    'data_logger': ['035-005'],
    'unit_retrieval': ['UNR-TRL'],
    'thermostat': ['082-019'],
    'expansion_valve': ['082-020'],
    'exhaust_egr': ['082-043'],
    'defrost': ['082-023'],
    'control_panel': ['082-015'],
    'remote_status_light': ['082-026'],
}
comp_to_cat = {}
for cat, codes in COMP_CATS.items():
    for code in codes:
        comp_to_cat[code] = cat


def get_policy(unit):
    return 120 if reefer_to_slxi.get(unit, 0) == 1 else 90


def get_season(dt):
    m = dt.month
    if m in [12, 1, 2]: return 'Winter'
    if m in [3, 4, 5]: return 'Spring'
    if m in [6, 7, 8]: return 'Summer'
    return 'Fall'


def get_alarm_severity_score(alarm_str):
    """Parse multi-value alarm codes and return (red_count, yellow_count, green_count)."""
    red, yellow, green = 0, 0, 0
    if not alarm_str or alarm_str == 'nan' or alarm_str == '':
        return red, yellow, green
    # Split on newlines, commas, spaces
    codes = re.split(r'[\n\r,\s]+', str(alarm_str))
    for code in codes:
        code = code.strip()
        if not code:
            continue
        sev = alarm_severity_map.get(code, '')
        if sev == 'Red':
            red += 1
        elif sev == 'Yellow':
            yellow += 1
        elif sev == 'Green':
            green += 1
    return red, yellow, green


def interpolate_engine_hours(readings, trip_date):
    """Interpolate engine hours at trip_date from sorted (date, reading) pairs."""
    if not readings:
        return np.nan, np.nan

    # Filter to readings before trip
    before = [(d, r) for d, r in readings if d < trip_date]
    if not before:
        return np.nan, np.nan

    last_date, last_reading = before[-1]
    hours_at_trip = last_reading  # best estimate: last known reading

    # Compute accumulation rate (hours per day) from last 2+ readings
    if len(before) >= 2:
        first_date, first_reading = before[0]
        days_span = (last_date - first_date).total_seconds() / 86400
        if days_span > 0:
            rate = (last_reading - first_reading) / days_span
        else:
            rate = np.nan
    else:
        rate = np.nan

    return hours_at_trip, rate


# ======================================================================
# BUILD FEATURES
# ======================================================================
print('\nComputing features for each trip...')

rows = []
for idx, trip in rail.iterrows():
    if idx % 5000 == 0:
        print(f'  Processing trip {idx}/{len(rail)}...')

    trip_date = trip['Leg_Start_Date1']
    trip_end = trip['Leg_End_Date1']
    unit = trip['reefer_unit']
    trailer = trip['Leg_Trailer1']
    policy = get_policy(unit)
    container_num = reefer_to_container.get(unit, trailer)

    row = {}

    # ---- IDENTIFIERS ----
    row['manifest'] = trip.get('Manifest1', '')
    row['leg'] = trip.get('Leg1', '')
    row['trailer'] = trailer
    row['reefer_unit'] = unit
    row['trip_start'] = trip_date
    row['trip_end'] = trip_end

    # ---- TRIP CONTEXT ----
    row['route'] = str(trip.get('Lane', ''))
    row['origin'] = str(trip.get('Start_Company1', ''))
    row['destination'] = str(trip.get('End_Company1', ''))
    row['leg_miles'] = trip.get('Leg_Miles', np.nan)
    row['total_miles'] = trip.get('Total Miles', np.nan)
    row['service_type'] = str(trip.get('Service_Type__Instructions_', ''))
    row['temperature'] = trip.get('Temperature__Run_at_Low_', np.nan)
    row['season'] = get_season(trip_date)
    row['month'] = trip_date.month
    row['day_of_week'] = trip_date.dayofweek

    if pd.notna(trip_end):
        row['trip_duration_days'] = (trip_end - trip_date).total_seconds() / 86400
    else:
        row['trip_duration_days'] = np.nan

    # ---- UNIT CHARACTERISTICS ----
    info = fleet_info.get(unit, {})
    row['is_slxi'] = info.get('is_slxi', 0)
    row['is_thermoking'] = info.get('is_thermoking', 0)
    row['is_carrier'] = info.get('is_carrier', 0)
    row['reefer_model'] = info.get('reefer_model', '')
    row['reefer_year'] = info.get('reefer_year', 0)
    row['pm_policy_days'] = policy

    reefer_yr = info.get('reefer_year', 0)
    container_yr = info.get('container_year', 0)
    row['reefer_age_at_trip'] = trip_date.year + (trip_date.month - 1) / 12 - reefer_yr if reefer_yr > 0 else np.nan
    row['container_age_at_trip'] = trip_date.year + (trip_date.month - 1) / 12 - container_yr if container_yr > 0 else np.nan

    install_dt = info.get('install_date')
    if pd.notna(install_dt):
        row['reefer_install_days_at_trip'] = (trip_date - install_dt).days
    else:
        row['reefer_install_days_at_trip'] = np.nan

    # ---- PM COMPLIANCE ----
    pms = pm_by_unit.get(unit, [])
    pms_before = [p for p in pms if p < trip_date]

    if pms_before:
        row['days_since_last_pm'] = (trip_date - pms_before[-1]).days
        row['is_pm_overdue'] = 1 if row['days_since_last_pm'] > policy else 0
        row['days_pm_overdue'] = max(0, row['days_since_last_pm'] - policy)
    else:
        row['days_since_last_pm'] = np.nan
        row['is_pm_overdue'] = np.nan
        row['days_pm_overdue'] = np.nan

    if len(pms_before) >= 2:
        gaps = [(pms_before[i+1] - pms_before[i]).days for i in range(len(pms_before)-1)]
        on_time = sum(1 for g in gaps if g <= policy)
        late = len(gaps) - on_time
        row['pm_on_time_count'] = on_time
        row['pm_late_count'] = late
        row['pm_total_intervals'] = len(gaps)
        row['pm_compliance_ratio'] = on_time / len(gaps)
        row['pm_gap_mean'] = np.mean(gaps)
        row['pm_gap_std'] = np.std(gaps)
        row['pm_gap_cv'] = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
        row['pm_gap_max'] = max(gaps)
        row['pm_gap_min'] = min(gaps)
    else:
        for col in ['pm_on_time_count', 'pm_late_count', 'pm_total_intervals',
                     'pm_compliance_ratio', 'pm_gap_mean', 'pm_gap_std',
                     'pm_gap_cv', 'pm_gap_max', 'pm_gap_min']:
            row[col] = np.nan

    # ---- MAJOR PM TYPES (v3 new) ----
    unit_major_pms = major_pm_by_unit.get(unit, {})
    for pm_type in ['major_service', 'starter_replacement', 'alternator_replacement',
                    'battery_replacement', 'fuel_polishing']:
        dates = unit_major_pms.get(pm_type, [])
        dates_before = [d for d in dates if d < trip_date]
        if dates_before:
            row[f'days_since_{pm_type}'] = (trip_date - dates_before[-1]).days
            row[f'has_had_{pm_type}'] = 1
        else:
            row[f'days_since_{pm_type}'] = np.nan
            row[f'has_had_{pm_type}'] = 0

    # ---- REEFER REPAIR FEATURES ----
    cutoff_30 = trip_date - pd.Timedelta(days=30)
    cutoff_90 = trip_date - pd.Timedelta(days=90)
    cutoff_180 = trip_date - pd.Timedelta(days=180)
    cutoff_365 = trip_date - pd.Timedelta(days=365)

    reps = repairs_by_unit.get(unit, [])
    reps_before = [r for r in reps if r['date'] < trip_date]
    nonpm_before = [r for r in reps_before if r['repreason'] != 'PM']
    pm_before = [r for r in reps_before if r['repreason'] == 'PM']
    reps_90d = [r for r in reps_before if r['date'] >= cutoff_90]
    reps_prior_90d = [r for r in reps_before if r['date'] < cutoff_90 and r['date'] >= cutoff_180]
    nonpm_90d = [r for r in nonpm_before if r['date'] >= cutoff_90]

    row['repair_cum_cost'] = sum(r['total_cost'] for r in reps_before)
    row['repair_cum_count'] = len(reps_before)
    row['nonpm_cum_cost'] = sum(r['total_cost'] for r in nonpm_before)
    row['nonpm_cum_count'] = len(nonpm_before)

    for window_name, cutoff in [('30d', cutoff_30), ('90d', cutoff_90),
                                 ('180d', cutoff_180), ('365d', cutoff_365)]:
        w_all = [r for r in reps_before if r['date'] >= cutoff]
        w_nonpm = [r for r in nonpm_before if r['date'] >= cutoff]
        row[f'repair_cost_{window_name}'] = sum(r['total_cost'] for r in w_all)
        row[f'repair_count_{window_name}'] = len(w_all)
        row[f'nonpm_cost_{window_name}'] = sum(r['total_cost'] for r in w_nonpm)
        row[f'nonpm_count_{window_name}'] = len(w_nonpm)

    # ---- ACTIONABLE REPAIR FEATURES ----
    # Repair acceleration: last 90d count vs prior 90d (speeding up?)
    count_last_90 = len(reps_90d)
    count_prior_90 = len(reps_prior_90d)
    row['repair_acceleration'] = count_last_90 / max(count_prior_90, 1) if count_prior_90 > 0 else (float(count_last_90) if count_last_90 > 0 else np.nan)

    # Repeat same repair in 90d (same REPREASON 2+ times = fix didn't hold)
    reasons_90d = [r['repreason'] for r in nonpm_90d]
    reason_counts = Counter(reasons_90d)
    row['repeat_repairs_90d'] = sum(1 for c in reason_counts.values() if c >= 2)
    row['max_repeat_count_90d'] = max(reason_counts.values()) if reason_counts else 0

    # GENREP to PM ratio (reactive vs preventive)
    row['genrep_to_pm_ratio'] = len(nonpm_before) / max(len(pm_before), 1)

    # Days since ANY shop visit (repair or PM)
    if reps_before:
        row['days_since_any_shop_visit'] = (trip_date - reps_before[-1]['date']).days
    else:
        row['days_since_any_shop_visit'] = np.nan

    # Unique repair reasons in 90d (problem diversity)
    row['unique_repair_reasons_90d'] = len(set(reasons_90d))

    # Shop bouncing in 365d
    shops_365 = set(r['shopid'] for r in reps_before if r['date'] >= cutoff_365)
    row['unique_shops_365d'] = len(shops_365)

    # ---- REPAIR TYPE FEATURES ----
    cat_counts_cum = defaultdict(int)
    cat_counts_90 = defaultdict(int)
    cat_counts_365 = defaultdict(int)
    unique_codes_cum = set()
    unique_codes_365 = set()
    unique_shops = set()
    has_dor = 0
    has_dor_90 = 0
    has_accident = 0
    has_warranty = 0

    for r in reps_before:
        unique_shops.add(r['shopid'])
        if r['repreason'] == 'ACC':
            has_accident = 1
        if r['repreason'] == 'WARRANTY':
            has_warranty = 1
        for cc in r['compcodes']:
            cat = comp_to_cat.get(cc, 'other')
            cat_counts_cum[cat] += 1
            unique_codes_cum.add(cc)
            if r['date'] >= cutoff_365:
                cat_counts_365[cat] += 1
                unique_codes_365.add(cc)
            if r['date'] >= cutoff_90:
                cat_counts_90[cat] += 1
            if cc == 'DOR-DIA':
                has_dor = 1
                if r['date'] >= cutoff_90:
                    has_dor_90 = 1

    row['unique_compcodes_cum'] = len(unique_codes_cum)
    row['unique_compcodes_365d'] = len(unique_codes_365)
    row['unique_shops'] = len(unique_shops)
    row['has_dor'] = has_dor
    row['has_dor_90d'] = has_dor_90
    row['has_accident'] = has_accident
    row['has_warranty'] = has_warranty

    for cat in COMP_CATS.keys():
        row[f'{cat}_cum'] = cat_counts_cum.get(cat, 0)
        row[f'{cat}_90d'] = cat_counts_90.get(cat, 0)
        row[f'{cat}_365d'] = cat_counts_365.get(cat, 0)

    # ---- PRE-TRIP REPAIR PATTERN FEATURES ----
    # Derived from repair pattern analysis — these capture the 14-day
    # pre-trip repair context that showed 2-5x lift over baseline.
    cutoff_14 = trip_date - pd.Timedelta(days=14)
    reps_14d = [r for r in reps_before if r['date'] >= cutoff_14]

    # has_dor_14d: Door/In-Out repair in 14 days before trip (14.4% rate, 2.85x lift)
    row['has_dor_14d'] = 1 if any(
        'DOR' in cc for r in reps_14d for cc in r['compcodes']
    ) else 0

    # has_electrical_14d: Electrical/Charging repair in 14 days (15.0% rate, 2.97x lift)
    row['has_electrical_14d'] = 1 if any(
        cc.startswith('035') for r in reps_14d for cc in r['compcodes']
    ) else 0

    # has_starter_14d: Starter/Cranking repair in 14 days (18.75% rate, 3.70x lift)
    row['has_starter_14d'] = 1 if any(
        cc.startswith('031') for r in reps_14d for cc in r['compcodes']
    ) else 0

    # days_since_last_repair: days from most recent repair to trip start
    if reps_before:
        row['days_since_last_repair'] = (trip_date - reps_before[-1]['date']).days
    else:
        row['days_since_last_repair'] = np.nan

    # diag_no_fix_14d: last work order in 14d was diagnosis-only, no component fix
    # (10.7% rate, 2.11x lift — unit sent out without real repair)
    if reps_14d:
        last_14d = max(reps_14d, key=lambda r: r['date'])
        has_diag = any(cc.endswith('-DIA') or cc == '082' for cc in last_14d['compcodes'])
        has_fix = any(
            not cc.endswith('-DIA') and not cc.startswith('000') and cc != '082'
            for cc in last_14d['compcodes']
        )
        row['diag_no_fix_14d'] = 1 if has_diag and not has_fix else 0
    else:
        row['diag_no_fix_14d'] = 0

    # n_repair_groups_14d: distinct component group prefixes in 14-day window
    # (combos of 2+ groups reach 15-25% failure rate)
    groups_14d = set()
    for r in reps_14d:
        for cc in r['compcodes']:
            prefix = cc.split('-')[0] if '-' in cc else cc[:3]
            groups_14d.add(prefix)
    row['n_repair_groups_14d'] = len(groups_14d)

    # has_repeat_repair_30d: same component group repaired 2+ times in 30 days
    # (repeat DOR = 12.3%, repeat electrical = 12.5%)
    reps_30d = [r for r in reps_before if r['date'] >= cutoff_30]
    group_counts_30d = Counter()
    for r in reps_30d:
        for cc in r['compcodes']:
            prefix = cc.split('-')[0] if '-' in cc else cc[:3]
            if prefix != '000':  # exclude PM
                group_counts_30d[prefix] += 1
    row['has_repeat_repair_30d'] = 1 if any(c >= 2 for c in group_counts_30d.values()) else 0

    # repair_cost_14d: total repair cost in 14-day window (severity proxy)
    row['repair_cost_14d'] = sum(r['total_cost'] for r in reps_14d)

    # ================================================================
    # v3 NEW FEATURES
    # ================================================================

    # ---- TELEMATICS EVENT HISTORY (v3 new) ----
    unit_events = telem_by_unit.get(unit, [])
    events_before = [e for e in unit_events if e['date'] < trip_date]

    # Counts by type
    shutdowns_before = [e for e in events_before if e['type'] == 'SHUTDOWN']
    alarms_before = [e for e in events_before if e['type'] == 'ALARM']
    low_fuel_before = [e for e in events_before if e['type'] == 'LOW_FUEL']
    restarts_before = [e for e in events_before if e['type'] == 'RESTARTED']
    not_reporting_before = [e for e in events_before if e['type'] == 'NOT_REPORTING']

    row['telem_shutdowns_total'] = len(shutdowns_before)
    row['telem_alarms_total'] = len(alarms_before)
    row['telem_low_fuel_total'] = len(low_fuel_before)
    row['telem_restarts_total'] = len(restarts_before)
    row['telem_not_reporting_total'] = len(not_reporting_before)
    row['telem_events_total'] = len(events_before)

    # Windowed counts (90d, 180d)
    for window_name, cutoff in [('90d', cutoff_90), ('180d', cutoff_180)]:
        w_events = [e for e in events_before if e['date'] >= cutoff]
        row[f'telem_shutdowns_{window_name}'] = sum(1 for e in w_events if e['type'] == 'SHUTDOWN')
        row[f'telem_alarms_{window_name}'] = sum(1 for e in w_events if e['type'] == 'ALARM')
        row[f'telem_low_fuel_{window_name}'] = sum(1 for e in w_events if e['type'] == 'LOW_FUEL')
        row[f'telem_events_{window_name}'] = len(w_events)

    # Recency features
    if shutdowns_before:
        row['days_since_last_telem_shutdown'] = (trip_date - shutdowns_before[-1]['date']).total_seconds() / 86400
    else:
        row['days_since_last_telem_shutdown'] = np.nan

    if alarms_before:
        row['days_since_last_alarm'] = (trip_date - alarms_before[-1]['date']).total_seconds() / 86400
    else:
        row['days_since_last_alarm'] = np.nan

    if low_fuel_before:
        row['days_since_last_low_fuel'] = (trip_date - low_fuel_before[-1]['date']).total_seconds() / 86400
    else:
        row['days_since_last_low_fuel'] = np.nan

    if events_before:
        row['days_since_last_telem_event'] = (trip_date - events_before[-1]['date']).total_seconds() / 86400
    else:
        row['days_since_last_telem_event'] = np.nan

    # ---- BATTERY VOLTAGE & FUEL % (v3 new) ----
    # Last known battery voltage before trip
    bv_readings = [(e['date'], e['battery']) for e in events_before if pd.notna(e['battery'])]
    if bv_readings:
        bv_readings.sort(key=lambda x: x[0])
        row['last_battery_voltage'] = bv_readings[-1][1]
        # Min battery voltage in last 90 days
        bv_90d = [bv for d, bv in bv_readings if d >= cutoff_90]
        row['min_battery_voltage_90d'] = min(bv_90d) if bv_90d else np.nan
        row['battery_below_12v'] = 1 if row['last_battery_voltage'] < 12.0 else 0
    else:
        row['last_battery_voltage'] = np.nan
        row['min_battery_voltage_90d'] = np.nan
        row['battery_below_12v'] = np.nan

    # Last known fuel %
    fuel_readings = [(e['date'], e['fuel']) for e in events_before if pd.notna(e['fuel'])]
    if fuel_readings:
        fuel_readings.sort(key=lambda x: x[0])
        row['last_fuel_pct'] = fuel_readings[-1][1]
    else:
        row['last_fuel_pct'] = np.nan

    # ---- ALARM SEVERITY (v3 new) ----
    alarm_red_total = 0
    alarm_yellow_total = 0
    alarm_green_total = 0
    alarm_red_90d = 0
    alarm_yellow_90d = 0

    for e in events_before:
        if e['alarm']:
            r_c, y_c, g_c = get_alarm_severity_score(e['alarm'])
            alarm_red_total += r_c
            alarm_yellow_total += y_c
            alarm_green_total += g_c
            if e['date'] >= cutoff_90:
                alarm_red_90d += r_c
                alarm_yellow_90d += y_c

    row['alarm_red_total'] = alarm_red_total
    row['alarm_yellow_total'] = alarm_yellow_total
    row['alarm_green_total'] = alarm_green_total
    row['alarm_red_90d'] = alarm_red_90d
    row['alarm_yellow_90d'] = alarm_yellow_90d
    # Weighted severity score: red=3, yellow=2, green=1
    row['alarm_severity_score'] = alarm_red_total * 3 + alarm_yellow_total * 2 + alarm_green_total
    row['alarm_severity_score_90d'] = alarm_red_90d * 3 + alarm_yellow_90d * 2

    # ---- ENGINE HOURS (v3 new) ----
    readings = meter_by_unit.get(unit, [])
    eng_hours, eng_rate = interpolate_engine_hours(readings, trip_date)
    row['engine_hours_at_trip'] = eng_hours
    row['engine_hours_per_day'] = eng_rate

    # Hours since last PM (if we have both engine hours and last PM date)
    if pd.notna(eng_hours) and pms_before:
        last_pm_hours = interpolate_engine_hours(readings, pms_before[-1])[0]
        if pd.notna(last_pm_hours):
            row['engine_hours_since_last_pm'] = eng_hours - last_pm_hours
        else:
            row['engine_hours_since_last_pm'] = np.nan
    else:
        row['engine_hours_since_last_pm'] = np.nan

    # ---- CONTAINER REPAIRS (v3 new) ----
    creps = container_repairs_by_num.get(container_num, [])
    creps_before = [r for r in creps if r['date'] < trip_date]

    row['container_repair_cum_count'] = len(creps_before)
    row['container_repair_cum_cost'] = sum(r['total_cost'] for r in creps_before)
    row['container_repair_count_90d'] = sum(1 for r in creps_before if r['date'] >= cutoff_90)
    row['container_repair_cost_90d'] = sum(r['total_cost'] for r in creps_before if r['date'] >= cutoff_90)
    row['container_repair_count_365d'] = sum(1 for r in creps_before if r['date'] >= cutoff_365)
    row['container_repair_cost_365d'] = sum(r['total_cost'] for r in creps_before if r['date'] >= cutoff_365)

    if creps_before:
        row['days_since_last_container_repair'] = (trip_date - creps_before[-1]['date']).days
    else:
        row['days_since_last_container_repair'] = np.nan

    # ---- FUEL FILL HISTORY (v3 new) ----
    fills = fuel_by_container.get(container_num, [])
    fills_before = [(d, l) for d, l in fills if d < trip_date]

    row['fuel_fills_total'] = len(fills_before)

    fills_30d = [(d, l) for d, l in fills_before if d >= cutoff_30]
    fills_90d = [(d, l) for d, l in fills_before if d >= cutoff_90]
    row['fuel_fills_30d'] = len(fills_30d)
    row['fuel_fills_90d'] = len(fills_90d)

    if fills_before:
        liters = [l for _, l in fills_before]
        row['fuel_avg_liters_per_fill'] = np.mean(liters)
        row['days_since_last_fuel_fill'] = (trip_date - fills_before[-1][0]).total_seconds() / 86400
    else:
        row['fuel_avg_liters_per_fill'] = np.nan
        row['days_since_last_fuel_fill'] = np.nan

    if fills_90d:
        row['fuel_avg_liters_90d'] = np.mean([l for _, l in fills_90d])
    else:
        row['fuel_avg_liters_90d'] = np.nan

    # ---- AMBIENT WEATHER FEATURES (v3 new) ----
    origin = row['origin']
    destination = row['destination']
    wx = compute_transit_weather(origin, destination, trip_date, trip_end)
    row['origin_temp'] = wx['origin_temp']
    row['dest_temp'] = wx['dest_temp']
    row['transit_temp_swing'] = wx['transit_temp_swing']
    row['transit_min_temp'] = wx['transit_min_temp']
    row['transit_max_temp'] = wx['transit_max_temp']
    row['origin_dest_temp_diff'] = wx['origin_dest_temp_diff']
    row['extreme_cold_flag'] = wx['extreme_cold_flag']
    row['danger_zone_flag'] = wx['danger_zone_flag']

    # Heating vs cooling mode based on setpoint vs ambient
    setpoint = row.get('temperature', np.nan)
    if pd.notna(setpoint) and pd.notna(wx['origin_temp']):
        row['heating_mode'] = 1 if setpoint > wx['origin_temp'] else 0
    else:
        row['heating_mode'] = np.nan

    # ---- OPERATING MODE (v3 new) ----
    # Most recent operating mode from telematics before trip
    mode_events = [e for e in events_before if e['mode'] and e['mode'] not in ('nan', '', 'None')]
    if mode_events:
        last_mode = mode_events[-1]['mode'].strip().upper()
        row['last_mode_continuous'] = 1 if last_mode == 'C' else 0
        row['last_mode_cycle_sentry'] = 1 if last_mode == 'CS' else 0
        # Count mode history
        continuous_count = sum(1 for e in events_before if str(e['mode']).strip().upper() == 'C')
        cs_count = sum(1 for e in events_before if str(e['mode']).strip().upper() == 'CS')
        total_mode = continuous_count + cs_count
        row['continuous_mode_ratio'] = continuous_count / total_mode if total_mode > 0 else np.nan
    else:
        row['last_mode_continuous'] = np.nan
        row['last_mode_cycle_sentry'] = np.nan
        row['continuous_mode_ratio'] = np.nan

    # ---- SHUTDOWN LABEL ----
    manifest_str = str(trip.get('Manifest1', '')).split('.')[0]
    row['shutdown'] = 1 if manifest_str in shutdown_manifests else 0

    rows.append(row)

# ======================================================================
# OUTPUT
# ======================================================================
df = pd.DataFrame(rows)
df['trip_start'] = pd.to_datetime(df['trip_start'])
pre_filter = len(df)

# Keep only years that have telematics shutdown labels (Combined_Telematics file scope).
# A year needs a meaningful shutdown rate (>= 1%) to be considered as having
# telematics coverage. Years without coverage have 0% shutdowns and would
# dilute the training data with unlabelled negatives.
year_stats = df.groupby(df['trip_start'].dt.year)['shutdown'].agg(['sum', 'count', 'mean'])
valid_years = sorted(year_stats[year_stats['mean'] >= 0.01].index.tolist())
if not valid_years:
    # Fallback: keep all years if none meet threshold
    valid_years = sorted(df['trip_start'].dt.year.unique())
df = df[df['trip_start'].dt.year.isin(valid_years)].reset_index(drop=True)

print(f'\nFiltered {pre_filter} -> {len(df)} (years: {valid_years})')
print(f'Dataset shape: {df.shape}')
print(f'Shutdowns: {df["shutdown"].sum()} / {len(df)} ({100*df["shutdown"].mean():.2f}%)')
print(f'Unique units: {df["reefer_unit"].nunique()}')
print(f'Date range: {df["trip_start"].min()} to {df["trip_start"].max()}')

# Column summary
print(f'\nTotal columns: {len(df.columns)}')

# New feature coverage
new_features = {
    'Telematics events': ['telem_shutdowns_total', 'telem_alarms_total', 'telem_low_fuel_total'],
    'Battery voltage': ['last_battery_voltage', 'min_battery_voltage_90d'],
    'Alarm severity': ['alarm_red_total', 'alarm_severity_score'],
    'Engine hours': ['engine_hours_at_trip', 'engine_hours_per_day'],
    'Container repairs': ['container_repair_cum_count', 'container_repair_cum_cost'],
    'Fuel fills': ['fuel_fills_total', 'fuel_avg_liters_per_fill'],
    'Major PM types': ['has_had_major_service', 'has_had_battery_replacement'],
    'Weather/ambient': ['origin_temp', 'transit_temp_swing', 'origin_dest_temp_diff'],
    'Operating mode': ['last_mode_continuous', 'continuous_mode_ratio'],
    'Pre-trip repair': ['has_dor_14d', 'has_electrical_14d', 'has_starter_14d',
                        'days_since_last_repair', 'diag_no_fix_14d',
                        'n_repair_groups_14d', 'has_repeat_repair_30d', 'repair_cost_14d'],
}

print('\n' + '=' * 60)
print('NEW FEATURE COVERAGE (non-null %)')
print('=' * 60)
for group, cols in new_features.items():
    for col in cols:
        if col in df.columns:
            pct = 100 * df[col].notna().mean()
            print(f'  {group:25s}  {col:40s}  {pct:5.1f}%')

outpath = 'data/processed/training/trip_features_v3.csv'
df.to_csv(outpath, index=False)
print(f'\nSaved to {outpath}')
print(f'File size: {os.path.getsize(outpath) / 1024 / 1024:.1f} MB')
print(f'Total features: {len(df.columns) - 7} (excluding identifiers + label)')
