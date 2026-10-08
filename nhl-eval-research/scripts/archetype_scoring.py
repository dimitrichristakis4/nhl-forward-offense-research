import pandas as pd

from config import WORK
from name_utils import normalize_name

# Builds the five play style scores (shooter, playmaker, transition, forecheck/cycle, net front).
# Each stat is z-scored within season and position. Most stats are first blended with the
# team average as (raw + above team) / 2, which separates how the player plays from how his team plays.
# The "career_avg" columns only use seasons before the current one.

player_ixg_stints = pd.read_csv(WORK / "player_ixg_stints.csv")
player_stints = pd.read_csv(WORK / "player_stints.csv")
stint_baselines = pd.read_csv(WORK / "stint_baselines.csv")
team_ixg_baselines = pd.read_csv(WORK / "team_ixg_baselines.csv")

# stints under 10 tracked games are too noisy, drop them everywhere
player_stints = player_stints[player_stints["GP"] >= 10].reset_index(drop=True)
player_ixg_stints = player_ixg_stints[player_ixg_stints["a3z_gp"] >= 10].reset_index(drop=True)

# shots on goal per 60, blended with the team average
player_stints["player_key"] = player_stints["Player"].apply(normalize_name)

player_stints = player_stints.merge(
    stint_baselines[["coach_stint", "pos_group", "Shots On Goal/60"]].rename(columns={"Shots On Goal/60": "team_SOG_per60"}),
    on=["coach_stint", "pos_group"],
    how="left"
)

player_stints["SOG_per60_above_team"] = player_stints["Shots On Goal/60"] - player_stints["team_SOG_per60"]
player_stints["SOG_per60_normalized"] = (player_stints["SOG_per60_above_team"] + player_stints["Shots On Goal/60"]) / 2
player_stints["season"] = player_stints["coach_stint"].str.extract(r"(\d{4}-\d{2})$")

# high danger chances per 60, blended with the team average
player_ixg_stints["pos_group"] = player_ixg_stints["Position"].apply(lambda p: "D" if p == "D" else "F")

player_ixg_stints = player_ixg_stints.merge(
    team_ixg_baselines[["coach_stint", "pos_group", "iHDCF/60"]].rename(columns={"iHDCF/60": "team_iHDCF_per60"}),
    on=["coach_stint", "pos_group"],
    how="left"
)

player_ixg_stints["iHDCF_per60_above_team"] = player_ixg_stints["iHDCF/60"] - player_ixg_stints["team_iHDCF_per60"]
player_ixg_stints["iHDCF_per60_normalized"] = (player_ixg_stints["iHDCF_per60_above_team"] + player_ixg_stints["iHDCF/60"]) / 2

# games played turn into a weight: under 10 is zero, 10 to 19 scales up, 20 or more is full
def sample_weight(gp):
    return 0.0 if gp < 10 else (gp / 20.0 if gp < 20 else 1.0)

# z-scores a stat within each (season, position) group, using the weights for the group mean and std.
# career_avg_z is the weighted average z-score from all of the player's earlier seasons.
# It is built from a running total minus the current season, because a player can have
# several stints in one season (trade or coaching change).
def weighted_zscore_and_career_avg(df, value_col, weight_col, season_col, prefix):
    df = df.copy()

    def weighted_stats(g):
        wt = g[weight_col]
        val = g[value_col]
        mean = (wt * val).sum() / wt.sum()
        var = (wt * (val - mean) ** 2).sum() / wt.sum()
        return pd.Series({"pop_mean": mean, "pop_std": var ** 0.5})

    group_stats = df.groupby([season_col, "Position"]).apply(weighted_stats).reset_index()
    df = df.merge(group_stats, on=[season_col, "Position"], how="left")

    df[f"{prefix}_zscore"] = (df[value_col] - df["pop_mean"]) / df["pop_std"]
    df = df.drop(columns=["pop_mean", "pop_std"])

    z_weighted = df[f"{prefix}_zscore"] * df[weight_col]
    season_level = pd.DataFrame({
        "season_z_weighted_sum": z_weighted.groupby([df["player_key"], df[season_col]]).transform("sum"),
        "season_weight_sum": df[weight_col].groupby([df["player_key"], df[season_col]]).transform("sum"),
    })
    df = pd.concat([df, season_level], axis=1)

    season_totals = df[["player_key", season_col, "season_z_weighted_sum", "season_weight_sum"]].drop_duplicates().sort_values(["player_key", season_col])
    season_totals["cum_z_weighted_sum"] = season_totals.groupby("player_key")["season_z_weighted_sum"].cumsum()
    season_totals["cum_weight_sum"] = season_totals.groupby("player_key")["season_weight_sum"].cumsum()
    season_totals["prior_z_weighted_sum"] = season_totals["cum_z_weighted_sum"] - season_totals["season_z_weighted_sum"]
    season_totals["prior_weight_sum"] = season_totals["cum_weight_sum"] - season_totals["season_weight_sum"]
    season_totals[f"{prefix}_career_avg_z"] = season_totals["prior_z_weighted_sum"] / season_totals["prior_weight_sum"]

    df = df.drop(columns=["season_z_weighted_sum", "season_weight_sum"]).merge(
        season_totals[["player_key", season_col, f"{prefix}_career_avg_z"]],
        on=["player_key", season_col],
        how="left"
    )
    return df

# blends one stat with the team average from stint_baselines, returns the new column name
def blend_with_team(df, col, name):
    base = stint_baselines[["coach_stint", "pos_group", col]].rename(columns={col: f"team_{name}"})
    df = df.merge(base, on=["coach_stint", "pos_group"], how="left")
    df[f"{name}_above_team"] = df[col] - df[f"team_{name}"]
    df[f"{name}_normalized"] = (df[f"{name}_above_team"] + df[col]) / 2
    return df, f"{name}_normalized"

player_stints["sample_weight"] = player_stints["GP"].apply(sample_weight)
player_stints = weighted_zscore_and_career_avg(player_stints, "SOG_per60_normalized", "sample_weight", "season", "SOG_per60")

player_ixg_stints["sample_weight"] = player_ixg_stints["a3z_gp"].apply(sample_weight)
player_ixg_stints = weighted_zscore_and_career_avg(player_ixg_stints, "iHDCF_per60_normalized", "sample_weight", "Year", "iHDCF_per60")

# shooter: shots on goal and high danger chances
archetypes = player_stints.merge(
    player_ixg_stints[["player_key", "coach_stint", "iHDCF_per60_zscore", "iHDCF_per60_career_avg_z"]],
    on=["player_key", "coach_stint"],
    how="left"
)

archetypes["Shooter_score"] = archetypes[["SOG_per60_zscore", "iHDCF_per60_zscore"]].mean(axis=1)
archetypes["Shooter_career_avg"] = archetypes[["SOG_per60_career_avg_z", "iHDCF_per60_career_avg_z"]].mean(axis=1)

# playmaker: high danger passes (blended), offensive zone assists and low-to-high passes (raw)
archetypes, col = blend_with_team(archetypes, "HD_Passes/60", "HD_passes_per60")
archetypes = weighted_zscore_and_career_avg(archetypes, col, "sample_weight", "season", "HD_passes_per60")
archetypes = weighted_zscore_and_career_avg(archetypes, "OZ Assist/60", "sample_weight", "season", "OZ_assists_per60")
archetypes = weighted_zscore_and_career_avg(archetypes, "Low-to-High /60", "sample_weight", "season", "low_to_high_per60")

archetypes["Playmaker_score"] = archetypes[["HD_passes_per60_zscore", "OZ_assists_per60_zscore", "low_to_high_per60_zscore"]].mean(axis=1)
archetypes["Playmaker_career_avg"] = archetypes[["HD_passes_per60_career_avg_z", "OZ_assists_per60_career_avg_z", "low_to_high_per60_career_avg_z"]].mean(axis=1)

# transition: neutral and defensive zone assists, clean exits, entries with possession (all blended),
# retrievals leading to exits (raw)
for stat, name in [("NZ Assist /60", "NZ_assist_per60"),
                   ("DZ Assist /60", "DZ_assist_per60"),
                   ("Clean Exit%", "clean_exit_pct"),
                   ("Entry w/possesion%", "entry_possession_pct")]:
    archetypes, col = blend_with_team(archetypes, stat, name)
    archetypes = weighted_zscore_and_career_avg(archetypes, col, "sample_weight", "season", name)

archetypes = weighted_zscore_and_career_avg(archetypes, "Retrievals Leading to Exits Rate", "sample_weight", "season", "retrievals_to_exits_rate")

archetypes["Transition_score"] = archetypes[["NZ_assist_per60_zscore", "DZ_assist_per60_zscore", "clean_exit_pct_zscore", "entry_possession_pct_zscore", "retrievals_to_exits_rate_zscore"]].mean(axis=1)
archetypes["Transition_career_avg"] = archetypes[["NZ_assist_per60_career_avg_z", "DZ_assist_per60_career_avg_z", "clean_exit_pct_career_avg_z", "entry_possession_pct_career_avg_z", "retrievals_to_exits_rate_career_avg_z"]].mean(axis=1)

# forecheck and cycle: forecheck pressures and recoveries (blended)
for stat, name in [("Forecheck Pressures/60", "forecheck_pressures_per60"),
                   ("Recoveries/60", "recoveries_per60")]:
    archetypes, col = blend_with_team(archetypes, stat, name)
    archetypes = weighted_zscore_and_career_avg(archetypes, col, "sample_weight", "season", name)

archetypes["Forecheck_Cycle_score"] = archetypes[["forecheck_pressures_per60_zscore", "recoveries_per60_zscore"]].mean(axis=1)
archetypes["Forecheck_Cycle_career_avg"] = archetypes[["forecheck_pressures_per60_career_avg_z", "recoveries_per60_career_avg_z"]].mean(axis=1)

# net front: deflections, deflection assists, rebounds (blended)
for stat, name in [("Deflections/60", "deflections_per60"),
                   ("Deflection Assists/60", "deflection_assists_per60"),
                   ("Rebounds/60", "rebounds_per60")]:
    archetypes, col = blend_with_team(archetypes, stat, name)
    archetypes = weighted_zscore_and_career_avg(archetypes, col, "sample_weight", "season", name)

archetypes["Net_Front_score"] = archetypes[["deflections_per60_zscore", "deflection_assists_per60_zscore", "rebounds_per60_zscore"]].mean(axis=1)
archetypes["Net_Front_career_avg"] = archetypes[["deflections_per60_career_avg_z", "deflection_assists_per60_career_avg_z", "rebounds_per60_career_avg_z"]].mean(axis=1)
