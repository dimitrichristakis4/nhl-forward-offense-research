import pandas as pd

from config import NST, WORK, SEASONS
from name_utils import normalize_name, normalize_team, resolve_player_key

# individual expected goals and high danger chances per 60 for each coach stint,
# plus a team average for each stint

season_map = {s: '20' + s[:2] + '-' + s[3:] for s in SEASONS}

frames = [pd.read_csv(NST / "ixg" / f"{s}.csv").assign(Year=y) for s, y in season_map.items()]
ixg = pd.concat(frames, ignore_index=True)
ixg['player_key'] = ixg.apply(lambda r: resolve_player_key(r['Player'], r['Position']), axis=1)

# take position from player_stints.csv. This file sometimes lists two positions for one
# player-season ("C, R"), which would drop real players from the team averages
player_stints_pos = pd.read_csv(WORK / "player_stints.csv")
player_stints_pos['player_key'] = player_stints_pos['Player'].apply(normalize_name)
position_by_player = player_stints_pos.groupby('player_key')['Position'].agg(lambda s: s.mode().iloc[0])

# Bolduc played center in St. Louis and wing after the trade to Montreal in 2025-26
POSITION_OVERRIDES = {
    ('zac bolduc', '2025-26'): 'W',
}

ixg['Position'] = ixg['player_key'].map(position_by_player)
ixg['Position'] = ixg.apply(lambda r: POSITION_OVERRIDES.get((r['player_key'], r['Year']), r['Position']), axis=1)
ixg = ixg[ixg['Position'].notna()].copy()

adj = ixg['TOI'] / 60
ixg['ixG/60'] = ixg['ixG'] / adj
ixg['Goals/60'] = ixg['Goals'] / adj
ixg['shooting_talent/60'] = (ixg['Goals'] - ixg['ixG']) / adj
ixg['ixG/shot'] = ixg['ixG'] / ixg['Shots'].replace(0, pd.NA)
ixg['iHDCF/60'] = ixg['iHDCF'] / adj

a3z = pd.read_csv(WORK / "a3z_game_stats_with_coaches.csv", low_memory=False)
a3z = a3z[a3z['Pos.'] != 'G'].copy()
a3z['Team'] = a3z['Team'].str.upper().apply(normalize_team)
a3z['player_key'] = a3z['Player'].apply(normalize_name)

a3z_team = a3z.groupby(['player_key', 'Team', 'Year'])['Game'].nunique().reset_index().rename(columns={'Game': 'a3z_gp'})
a3z_stint = a3z.groupby(['player_key', 'coach_stint', 'Team', 'Year'])['Game'].nunique().reset_index().rename(columns={'Game': 'a3z_gp'})

# NST has one row per player-season with every team listed. Split it across teams, then across
# coach stints, using the share of tracked games. The rates stay the same, only the games get split.
rate_cols = ['ixG/60', 'Goals/60', 'shooting_talent/60', 'ixG/shot', 'iHDCF/60']
split_rows = []

for _, row in ixg.iterrows():
    teams = [normalize_team(t.strip()) for t in str(row['Team']).split(',')]
    team_gps = {t: a3z_team[(a3z_team['player_key'] == row['player_key']) & (a3z_team['Team'] == t) & (a3z_team['Year'] == row['Year'])]['a3z_gp'].sum() for t in teams}
    total_team = sum(team_gps.values())

    for t in teams:
        team_ratio = team_gps[t] / total_team if total_team > 0 else 1 / len(teams)
        stints = a3z_stint[(a3z_stint['player_key'] == row['player_key']) & (a3z_stint['Team'] == t) & (a3z_stint['Year'] == row['Year'])]
        total_stint = stints['a3z_gp'].sum()

        if len(stints) == 0 or total_stint == 0:
            continue
        else:
            for _, s in stints.iterrows():
                split_rows.append({'Player': row['Player'], 'player_key': row['player_key'], 'Year': row['Year'], 'Position': row['Position'], 'Team': t, 'coach_stint': s['coach_stint'], 'split_gp': row['GP'] * team_ratio * (s['a3z_gp'] / total_stint), 'a3z_gp': s['a3z_gp'], **{c: row[c] for c in rate_cols}})

splits = pd.DataFrame(split_rows)

splits['pos_group'] = splits['Position'].apply(lambda p: 'D' if p == 'D' else 'F')

for col in rate_cols:
    splits[col + '_weighted'] = splits[col] * splits['split_gp']

gp_sum = splits.groupby(['coach_stint', 'Team', 'Year', 'pos_group'])['split_gp'].sum().reset_index()
weighted_sums = splits.groupby(['coach_stint', 'Team', 'Year', 'pos_group'])[[c + '_weighted' for c in rate_cols]].sum().reset_index()

team_baselines = gp_sum.merge(weighted_sums, on=['coach_stint', 'Team', 'Year', 'pos_group'])
for col in rate_cols:
    team_baselines[col] = team_baselines[col + '_weighted'] / team_baselines['split_gp']
    team_baselines = team_baselines.drop(columns=[col + '_weighted'])

team_baselines.to_csv(WORK / "team_ixg_baselines.csv", index=False)

splits[['Player', 'player_key', 'coach_stint', 'Team', 'Year', 'Position', 'split_gp', 'a3z_gp'] + rate_cols].to_csv(WORK / "player_ixg_stints.csv", index=False)
