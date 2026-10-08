import pandas as pd
import numpy as np

from config import ROOT, PRIVATE, WORK
from name_utils import normalize_team

# tags every All Three Zones game row with the coach who was behind the bench that day

ct = pd.read_csv(ROOT / "data" / "coach_tenures.csv")
ct['start_date'] = pd.to_datetime(ct['start_date'])
ct['end_date'] = pd.to_datetime(ct['end_date'])

gs = pd.read_excel(PRIVATE / "a3z_players_21_26.xlsx", sheet_name='Game Stats')

# regular season only, playoff years end in 'p'
gs = gs[~gs['Year'].astype(str).str.endswith('p')].copy()

# typo in the source data
gs['Game'] = gs['Game'].str.replace('12//7/2021', '12/7/2021', regex=False)

# Game looks like "1/1/2022 BOS vs. BUF"
gs['game_date'] = gs['Game'].str.extract(r'^(\d{1,2}/\d{1,2}/\d{4})')[0]
gs['game_date'] = pd.to_datetime(gs['game_date'], format='%m/%d/%Y')

gs['team_norm'] = gs['Team'].apply(normalize_team)

missing_dates = gs['game_date'].isna().sum()
if missing_dates > 0:
    print(f'{missing_dates} rows with no parseable game date')

# coach stints per (team, season), sorted by start date
tenure_lookup = {}
for _, row in ct.iterrows():
    key = (row['team'], row['season_label'])
    if key not in tenure_lookup:
        tenure_lookup[key] = []
    tenure_lookup[key].append((row['start_date'], row['end_date'], row['coach'], row['season_label']))

for key in tenure_lookup:
    tenure_lookup[key].sort(key=lambda x: x[0])

coach_names = []
coach_stints = []

for _, row in gs.iterrows():
    team = row['team_norm']
    date = row['game_date']
    season = row['Year']

    if pd.isna(date):
        coach_names.append(np.nan)
        coach_stints.append(np.nan)
        continue

    # fill in the season from the date if it's missing
    if pd.isna(season):
        yr = date.year
        season = f'{yr-1}-{str(yr)[2:]}' if date.month < 9 else f'{yr}-{str(yr+1)[2:]}'

    key = (team, season)
    stints = tenure_lookup.get(key)

    if not stints:
        coach_names.append(np.nan)
        coach_stints.append(np.nan)
        continue

    matched = False
    for start, end, coach, slabel in stints:
        if start <= date <= end:
            coach_names.append(coach)
            coach_stints.append(f'{coach} {team} {slabel}')
            matched = True
            break

    if not matched:
        # date falls just outside every stint, use the closest one
        closest = min(stints, key=lambda s: min(abs((date - s[0]).days), abs((date - s[1]).days)))
        coach_names.append(closest[2])
        coach_stints.append(f'{closest[2]} {team} {closest[3]}')

gs['coach'] = coach_names
gs['coach_stint'] = coach_stints

gs.drop(columns=['game_date', 'team_norm'], inplace=True)

print(f'{gs["coach_stint"].nunique()} coach stints, {gs["coach"].isna().sum()} unmatched rows')

out = WORK / "a3z_game_stats_with_coaches.csv"
gs.to_csv(out, index=False)
print(f'saved to {out}')
