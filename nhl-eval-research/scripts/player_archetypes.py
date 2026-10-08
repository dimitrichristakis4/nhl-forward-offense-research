import pandas as pd

from config import WORK

# turns the game-level All Three Zones rows into per-60 stats for every player stint,
# plus a team average for each coach stint (forwards and defense kept separate)

df = pd.read_csv(WORK / "a3z_game_stats_with_coaches.csv")
df['Team'] = df['Team'].str.upper()
df = df[df['Pos.'] != 'G']

# these players have forward and defense games mixed up somewhere in their career,
# so they are dropped completely
MIXED_POSITION_PLAYERS = {
    'Brendan Smith',
    'Caleb Jones',
    'Ian Moore',
    'Jacob MacDonald',
    'Joel Hanley',
    'Kurtis MacDermid',
    'Mark Friedman',
    'Oliver Kylington',
    'Troy Stecher',
    'Uvis Balinskis',
}
# player names in this file use a non-breaking space between first and last name
player_normalized = df['Player'].str.replace('\xa0', ' ', regex=False)
df = df[~player_normalized.isin(MIXED_POSITION_PLAYERS)]

df['pos_group'] = df['Pos.'].map({'C': 'F', 'L': 'F', 'R': 'F', 'F': 'F', 'D': 'D'})

# one position per player: D, C, or W for any other forward
specific_counts = df[df['Pos.'] != 'F'].groupby('Player')['Pos.'].value_counts().unstack(fill_value=0)
for letter in ['C', 'L', 'R', 'D']:
    if letter not in specific_counts.columns:
        specific_counts[letter] = 0

def resolve_position(row):
    forward = row['C'] + row['L'] + row['R']
    if row['D'] > forward:
        return 'D'
    if forward == 0:
        return 'W'
    if row['C'] > (row['L'] + row['R']):
        return 'C'
    return 'W'

player_position = specific_counts.apply(resolve_position, axis=1)
player_position = player_position.reindex(df['Player'].unique()).fillna('W')
df['Position'] = df['Player'].map(player_position)


# some column names have trailing spaces in the source file, keep them as is
cols = ['Shots On Goal', 'Home Plate ', 'Behind Net', '5v5 TOI',
        'Carries ', 'Zone Entries', 'Failed Entries',
        'Exits w/ Possession', 'Zone Exits', 'Failed Exit',
        'Shots off Rush', 'Assists off Rush',
        'Shots off Forecheck or Cycle', 'Shots off Cycle', 'Assists off Forecheck', 'Assists off Cycle',
        'OZ Assist', 'Low-to-High ', 'NZ Assist ', 'DZ Assist ', 'Forecheck Pressures', 'Recoveries', 'Retrievals Leading to Exits', 'Rebounds', 'Deflections', 'Rebound Assists', 'Deflection Assists',
        'Denials', 'Targets ', 'Carries  ', 'Carries w/ Chances', 'Dump-in Chances', 'DZ Retrievals']
player_stints = df.groupby(['Player', 'coach_stint', 'Team', 'pos_group', 'Position'])[cols].sum().reset_index()
gp = df.groupby(['Player', 'coach_stint', 'Team', 'pos_group', 'Position'])['Game'].nunique().rename('GP').reset_index()
player_stints = player_stints.merge(gp, on=['Player', 'coach_stint', 'Team', 'pos_group', 'Position'])
# the column with two trailing spaces is carries against
player_stints = player_stints.rename(columns={'Carries  ': 'carries_against'})

per_60_adjuster = player_stints['5v5 TOI'] / 60

player_stints['HD_Passes'] = player_stints['Home Plate '] + player_stints['Behind Net']
per_60_cols = player_stints[['Shots On Goal', 'HD_Passes', 'Shots off Rush', 'Assists off Rush',
        'Shots off Forecheck or Cycle', 'Shots off Cycle', 'Assists off Forecheck', 'Assists off Cycle',
        'OZ Assist', 'Low-to-High ', 'NZ Assist ', 'DZ Assist ', 'Forecheck Pressures', 'Recoveries', 'Rebounds', 'Deflections', 'Rebound Assists', 'Deflection Assists',
        'Denials']]
for col in per_60_cols:
    player_stints[col + '/60'] = player_stints[col] / per_60_adjuster

per_60_final_cols = [col + '/60' for col in per_60_cols]

player_stints['Entry w/possesion%'] = player_stints['Carries '] / (player_stints['Zone Entries'] + player_stints['Failed Entries'])
player_stints['Clean Exit%'] = player_stints['Exits w/ Possession'] / (player_stints['Zone Exits'] + player_stints['Failed Exit'])
player_stints['Entry w/possesion allowed %'] = player_stints['carries_against'] / (player_stints['Targets '])
player_stints.loc[player_stints['pos_group'] == 'F', 'Entry w/possesion allowed %'] = None

player_stints['Carries w/ Chances Rate'] = player_stints['Carries w/ Chances'] / player_stints['Carries ']
player_stints['Dumps'] = player_stints['Zone Entries'] - player_stints['Carries ']
player_stints['Dump-in Chance Rate'] = player_stints['Dump-in Chances'] / player_stints['Dumps']
player_stints['Retrievals Leading to Exits Rate'] = player_stints['Retrievals Leading to Exits'] / player_stints['DZ Retrievals']
player_stints.loc[player_stints['pos_group'] == 'F', 'Retrievals Leading to Exits Rate'] = None

final_cols = per_60_final_cols + ['Entry w/possesion%', 'Clean Exit%', 'Entry w/possesion allowed %', 'Carries w/ Chances Rate', 'Dump-in Chance Rate', 'Retrievals Leading to Exits Rate']

# team average for each stat = average of the players' values weighted by games played
for col in final_cols:
    player_stints[col + '_weighted'] = player_stints[col] * player_stints['GP']

weighted_cols = [col + '_weighted' for col in final_cols]

gp_sum = player_stints.groupby(['coach_stint', 'Team', 'pos_group'])['GP'].sum().reset_index()
weighted_sums = player_stints.groupby(['coach_stint', 'Team', 'pos_group'])[weighted_cols].sum().reset_index()

stint_baselines = gp_sum.merge(weighted_sums, on=['coach_stint', 'Team', 'pos_group'])
for col in final_cols:
    stint_baselines[col] = stint_baselines[col + '_weighted'] / stint_baselines['GP']
    stint_baselines = stint_baselines.drop(columns=[col + '_weighted'])

stint_baselines.to_csv(WORK / "stint_baselines.csv", index=False)

player_stints[['Player', 'coach_stint', 'Team', 'pos_group', 'Position', 'GP'] + final_cols].to_csv(WORK / "player_stints.csv", index=False)
