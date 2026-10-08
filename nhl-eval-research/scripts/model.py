import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from config import NST, PRIVATE, RESULTS, SEASONS
from name_utils import normalize_name, resolve_player_key, normalize_team
from archetype_scoring import archetypes as _archetypes, stint_baselines, team_ixg_baselines

FULL_NAME_TO_ABBR = {
    "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF", "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL", "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA", "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL", "Nashville Predators": "NSH", "New Jersey Devils": "NJD",
    "New York Islanders": "NYI", "New York Rangers": "NYR", "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA", "St. Louis Blues": "STL", "St Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR", "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
    "Utah Hockey Club": "UTA", "Arizona Coyotes ": "ARI",
}

def zscore_within(df, col, group_cols):
    g = df.groupby(group_cols)[col]
    return (df[col] - g.transform("mean")) / g.transform("std")

def weighted_avg_by_group(df, group_cols, weight_col, value_col):
    sub = df[group_cols + [weight_col, value_col]].dropna(subset=[value_col]).copy()
    sub["_w"] = sub[weight_col] * sub[value_col]
    g = sub.groupby(group_cols)
    return g["_w"].sum() / g[weight_col].sum()

# 1. style scores and team baselines from archetype_scoring.py
archetypes = _archetypes.copy()

DIMS = ["Shooter_career_avg", "Playmaker_career_avg", "Transition_career_avg",
        "Forecheck_Cycle_career_avg", "Net_Front_career_avg"]

# one row per player-season, keeping the stint with the most games
shooter_primary = (
    archetypes.sort_values("GP", ascending=False)
    .drop_duplicates(subset=["player_key", "season"], keep="first")
    [["player_key", "season", "Player", "coach_stint", "Team", "Position", "pos_group", "GP"] + DIMS]
    .copy()
)
shooter_primary["season2"] = shooter_primary["season"].str[-5:]

# 2. team style features for each coach stint and position group
sys_feat = stint_baselines[[
    "coach_stint", "pos_group", "Clean Exit%", "Entry w/possesion%",
    "Shots On Goal/60", "Shots off Forecheck or Cycle/60", "Shots off Rush/60"
]].merge(
    team_ixg_baselines[["coach_stint", "pos_group", "iHDCF/60"]],
    on=["coach_stint", "pos_group"], how="left"
)
sys_feat["season"] = sys_feat["coach_stint"].str.extract(r"(\d{4}-\d{2})$")

sys_feat["hdcf_per_sog"] = sys_feat["iHDCF/60"] / sys_feat["Shots On Goal/60"]
sys_feat["sog_z"] = zscore_within(sys_feat, "Shots On Goal/60", ["season", "pos_group"])
sys_feat["hdcf_ratio_z"] = zscore_within(sys_feat, "hdcf_per_sog", ["season", "pos_group"])
sys_feat["shot_efficiency"] = sys_feat[["sog_z", "hdcf_ratio_z"]].mean(axis=1)

sys_feat["style_ratio_raw"] = sys_feat["Shots off Forecheck or Cycle/60"] / sys_feat["Shots off Rush/60"]
sys_feat["style_ratio"] = zscore_within(sys_feat, "style_ratio_raw", ["season", "pos_group"])

sys_feat["clean_exit_z"] = zscore_within(sys_feat, "Clean Exit%", ["season", "pos_group"])
sys_feat["entry_poss_z"] = zscore_within(sys_feat, "Entry w/possesion%", ["season", "pos_group"])
sys_feat["possession_rate"] = sys_feat[["clean_exit_z", "entry_poss_z"]].mean(axis=1)

sys_feat_final = sys_feat[["coach_stint", "pos_group", "possession_rate", "shot_efficiency", "style_ratio"]]
panel = shooter_primary.merge(sys_feat_final, on=["coach_stint", "pos_group"], how="left")

# 3. xEVO per game (Evolving Hockey, even strength offense)
xevo = pd.read_csv(PRIVATE / "eh_xevo_xevd.csv")
xevo["player_key"] = xevo.apply(lambda r: resolve_player_key(r["Player"], r["Position"]), axis=1)
xevo["Team"] = xevo["Team"].apply(normalize_team)
xevo_primary = (
    xevo.sort_values("GP", ascending=False)
    .drop_duplicates(subset=["player_key", "Season"], keep="first")
    .copy()
)
xevo_primary["xEVO_per_gp"] = xevo_primary["xEVO_GAR"] / xevo_primary["GP"]
xevo_slim = xevo_primary[["player_key", "Season", "xEVO_per_gp"]].rename(columns={"Season": "season2"})
panel = panel.merge(xevo_slim, on=["player_key", "season2"], how="left")

# 4. team xGF/60 from the NST team tables
team_rows = []
for season2 in SEASONS:
    df = pd.read_csv(NST / "fenwick_team" / f"{season2}.csv")
    df["season2"] = season2
    df["team_abbr"] = df["Team"].map(FULL_NAME_TO_ABBR).apply(normalize_team)
    df["team_xgf_per60"] = df["xGF"] / (df["TOI"] / 60.0)
    team_rows.append(df[["season2", "team_abbr", "team_xgf_per60"]])
team_xgf = pd.concat(team_rows, ignore_index=True)

# 5. player on-ice xGF/60 at 5v5, blended with his team's number
onice = pd.read_csv(NST / "five_on_five_xg.csv")
onice["player_key"] = onice.apply(lambda r: resolve_player_key(r["Player"], r["Position"]), axis=1)
onice["team_abbr"] = onice["Team"].apply(normalize_team)
onice_primary = (
    onice.sort_values("GP", ascending=False)
    .drop_duplicates(subset=["player_key", "Season"], keep="first")
    .copy()
)
onice_slim = onice_primary[["player_key", "Season", "team_abbr", "xGF/60"]].rename(
    columns={"Season": "season2", "xGF/60": "player_xgf60"}
)
onice_slim = onice_slim.merge(team_xgf, on=["season2", "team_abbr"], how="left")
onice_slim["above_team_xgf60"] = onice_slim["player_xgf60"] - onice_slim["team_xgf_per60"]
onice_slim["team_norm_xgf60"] = (onice_slim["player_xgf60"] + onice_slim["above_team_xgf60"]) / 2

panel = panel.merge(
    onice_slim[["player_key", "season2", "player_xgf60", "team_norm_xgf60"]],
    on=["player_key", "season2"], how="left"
)

# 6. target: average of the two z-scores
panel["xevo_z"] = zscore_within(panel, "xEVO_per_gp", ["season2"])
panel["xgf60_z"] = zscore_within(panel, "team_norm_xgf60", ["season2"])
panel["blend_score"] = panel[["xevo_z", "xgf60_z"]].mean(axis=1)

# 7. flag players whose coach or team changed since last season, and add last season's target
panel["coach_team_id"] = panel["coach_stint"].str.replace(r"\s+\d{4}-\d{2}$", "", regex=True)
panel = panel.sort_values(["player_key", "season2"]).reset_index(drop=True)
panel["coach_team_prev"] = panel.groupby("player_key")["coach_team_id"].shift(1)
panel["is_switcher"] = (panel["coach_team_prev"].notna()) & (panel["coach_team_prev"] != panel["coach_team_id"])
panel["blend_prev"] = panel.groupby("player_key")["blend_score"].shift(1)

# 8. each forward's most used line (by time on ice) from the Evolving Hockey line combos
combo_frames = []
for season2 in SEASONS:
    df = pd.read_csv(PRIVATE / "eh_forward_lines" / f"{season2}.csv")
    df["season2"] = season2
    combo_frames.append(df)
combos = pd.concat(combo_frames, ignore_index=True)

# each line becomes three rows, one per player, pointing at the other two
long_rows = []
for _, r in combos.iterrows():
    names = [r["Player 1"], r["Player 2"], r["Player 3"]]
    for i in range(3):
        others = [names[j] for j in range(3) if j != i]
        long_rows.append({
            "player_key": normalize_name(names[i]),
            "season2": r["season2"],
            "linemate_a_name": others[0],
            "linemate_b_name": others[1],
            "TOI": r["TOI"],
        })
long_combo = pd.DataFrame(long_rows)

top_trio = (
    long_combo.sort_values("TOI", ascending=False)
    .drop_duplicates(subset=["player_key", "season2"], keep="first")
    .copy()
)
top_trio["linemate_a_key"] = top_trio["linemate_a_name"].apply(normalize_name)
top_trio["linemate_b_key"] = top_trio["linemate_b_name"].apply(normalize_name)

# how many earlier seasons the player and each linemate were already on the top line together
# (the current season is left out)
pair_history = pd.concat([
    top_trio[["player_key", "season2", "linemate_a_key"]].rename(columns={"linemate_a_key": "teammate_key"}),
    top_trio[["player_key", "season2", "linemate_b_key"]].rename(columns={"linemate_b_key": "teammate_key"}),
], ignore_index=True)
pair_history = pair_history.sort_values(["player_key", "teammate_key", "season2"])
pair_history["prior_pairings"] = pair_history.groupby(["player_key", "teammate_key"]).cumcount()

top_trio = top_trio.merge(
    pair_history.rename(columns={"teammate_key": "linemate_a_key", "prior_pairings": "linemate_a_prior_pairings"}),
    on=["player_key", "season2", "linemate_a_key"], how="left"
)
top_trio = top_trio.merge(
    pair_history.rename(columns={"teammate_key": "linemate_b_key", "prior_pairings": "linemate_b_prior_pairings"}),
    on=["player_key", "season2", "linemate_b_key"], how="left"
)

# 9. add each linemate's style scores
arche_lookup = shooter_primary[["player_key", "season2"] + DIMS]

for side in ["a", "b"]:
    renamed = arche_lookup.rename(columns={"player_key": f"linemate_{side}_key", **{d: f"linemate_{side}_{d}" for d in DIMS}})
    top_trio = top_trio.merge(renamed, on=[f"linemate_{side}_key", "season2"], how="left")

linemate_cols = (
    [f"linemate_a_{d}" for d in DIMS] + [f"linemate_b_{d}" for d in DIMS]
    + ["linemate_a_prior_pairings", "linemate_b_prior_pairings"]
)
panel = panel.merge(top_trio[["player_key", "season2"] + linemate_cols], on=["player_key", "season2"], how="left")

# 10. quality of competition (average strength of his opponents) and ice time per game
qoc = pd.read_csv(PRIVATE / "eh_quality_of_competition.csv")
qoc["player_key"] = qoc.apply(lambda r: resolve_player_key(r["Player"], r["Position"]), axis=1)
qoc_primary = (
    qoc.sort_values("GP", ascending=False)
    .drop_duplicates(subset=["player_key", "Season"], keep="first")
    .copy()
)
qoc_slim = qoc_primary[["player_key", "Season", "xGAR±/60", "TOI/GP"]].rename(
    columns={"Season": "season2", "xGAR±/60": "qoc_xgar", "TOI/GP": "toi_per_gp"}
)
panel = panel.merge(qoc_slim, on=["player_key", "season2"], how="left")

# ice time is only known in advance from last season, so z-score it and lag it by one season
panel["toi_gp_z"] = zscore_within(panel, "toi_per_gp", ["season2"])
panel["toi_gp_prev"] = panel.groupby("player_key")["toi_gp_z"].shift(1)

# 10b. age (the only column used from the RAPM file)
age_df = pd.read_csv(PRIVATE / "eh_rapm_ev.csv")
age_df["player_key"] = age_df.apply(lambda r: resolve_player_key(r["Player"], r["Position"]), axis=1)
age_primary = (
    age_df.sort_values("GP", ascending=False)
    .drop_duplicates(subset=["player_key", "Season"], keep="first")
    .copy()
)
age_slim = age_primary[["player_key", "Season", "Age"]].rename(columns={"Season": "season2", "Age": "age"})
panel = panel.merge(age_slim, on=["player_key", "season2"], how="left")

# 10c. zone start rates from the NST player tables
zone_frames = []
for season2 in SEASONS:
    df = pd.read_csv(NST / "fenwick_player" / f"{season2}.csv")
    df["season2"] = season2
    zone_frames.append(df)
zone_df = pd.concat(zone_frames, ignore_index=True)
zone_df["player_key"] = zone_df.apply(lambda r: resolve_player_key(r["Player"], r["Position"]), axis=1)
zone_primary = (
    zone_df.sort_values("GP", ascending=False)
    .drop_duplicates(subset=["player_key", "season2"], keep="first")
    .copy()
)
# the NST column names have non-breaking spaces in them
zone_total = zone_primary["Off.\xa0Zone Starts"] + zone_primary["Neu.\xa0Zone Starts"] + zone_primary["Def.\xa0Zone Starts"]
zone_primary["oz_start_rate"] = zone_primary["Off.\xa0Zone Starts"] / zone_total
zone_primary["dz_start_rate"] = zone_primary["Def.\xa0Zone Starts"] / zone_total
zone_slim = zone_primary[["player_key", "season2", "oz_start_rate", "dz_start_rate"]]
panel = panel.merge(zone_slim, on=["player_key", "season2"], how="left")

# 10d. defense partners: his top 6 D partners by time together, averaging their style scores
# weighted by that time. Only rows for the team he played the most games with are used.
dpartner_frames = []
for season2 in SEASONS:
    df = pd.read_csv(PRIVATE / "eh_forwards_and_d" / f"{season2}.csv")
    df["season2"] = season2
    dpartner_frames.append(df)
dpartner_raw = pd.concat(dpartner_frames, ignore_index=True)

dpartner_raw["player_key"] = dpartner_raw["Player"].apply(normalize_name)
dpartner_raw["teammate_key"] = dpartner_raw["Teammate"].apply(normalize_name)

dpartner_raw = dpartner_raw.merge(
    panel[["player_key", "season2", "Team"]].dropna(subset=["Team"]).drop_duplicates().rename(columns={"Team": "dominant_team"}),
    on=["player_key", "season2"], how="inner"
)
dpartner_raw = dpartner_raw[dpartner_raw["Team"] == dpartner_raw["dominant_team"]]

dpartner_top6 = (
    dpartner_raw.sort_values("TOI_tog", ascending=False)
    .groupby(["player_key", "season2"])
    .head(6)
    .copy()
)

arche_partner = arche_lookup.rename(columns={"player_key": "teammate_key", **{d: f"partner_{d}" for d in DIMS}})
dpartner_top6 = dpartner_top6.merge(arche_partner, on=["teammate_key", "season2"], how="left")

dpartner_composite = dpartner_top6[["player_key", "season2"]].drop_duplicates().reset_index(drop=True)
for dim in DIMS:
    result = weighted_avg_by_group(dpartner_top6, ["player_key", "season2"], "TOI_tog", f"partner_{dim}")
    result = result.rename(f"d_partner_{dim}").reset_index()
    dpartner_composite = dpartner_composite.merge(result, on=["player_key", "season2"], how="left")

panel = panel.merge(dpartner_composite, on=["player_key", "season2"], how="left")

# 11. features and target
FEATURES = (
    DIMS
    + ["possession_rate", "shot_efficiency", "style_ratio", "blend_prev", "toi_gp_prev", "age"]
    + [f"linemate_a_{d}" for d in DIMS]
    + [f"linemate_b_{d}" for d in DIMS]
    + ["linemate_a_prior_pairings", "linemate_b_prior_pairings"]
    + [f"d_partner_{d}" for d in DIMS]
    + ["qoc_xgar", "oz_start_rate", "dz_start_rate"]
)
TARGET = "blend_score"

# kept shallow and heavily regularized because there are only about 550 rows
MODEL_PARAMS = dict(
    max_depth=3, learning_rate=0.05, n_estimators=300,
    min_child_weight=15, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=5.0,
)

# forwards whose coach or team changed since last season, with every feature present
model_df = panel[(panel["pos_group"] == "F") & (panel["is_switcher"])].dropna(
    subset=[TARGET] + FEATURES
).copy()

def r2_score_simple(actual, pred):
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - ss_res / ss_tot

def run_cv(df, feats, target, seed=42):
    X = df[feats].values
    y = df[target].values
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2s, rmses, maes = [], [], []
    for train_idx, test_idx in kf.split(X):
        model = XGBRegressor(random_state=seed, **MODEL_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        actual = y[test_idx]
        r2s.append(r2_score_simple(actual, pred))
        rmses.append(np.sqrt(np.mean((actual - pred) ** 2)))
        maes.append(np.mean(np.abs(actual - pred)))
    return np.mean(r2s), np.mean(rmses), np.mean(maes)


if __name__ == "__main__":
    print(f"Forward switchers with all features: n={len(model_df)}")

    r2, rmse, mae = run_cv(model_df, FEATURES, TARGET)
    print(f"5-fold CV (seed 42): R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # one seed can be off by +/-0.02, so judge changes on the 20 seed average
    seed_r2s = np.array([run_cv(model_df, FEATURES, TARGET, seed=s)[0] for s in range(20)])
    print(f"20-seed CV: R2 mean={seed_r2s.mean():.4f}  std={seed_r2s.std():.4f}  "
          f"min={seed_r2s.min():.4f}  max={seed_r2s.max():.4f}")

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame({"seed": range(20), "r2": seed_r2s.round(4)}).to_csv(RESULTS / "seed_r2.csv", index=False)

    final_model = XGBRegressor(random_state=42, **MODEL_PARAMS)
    final_model.fit(model_df[FEATURES].values, model_df[TARGET].values)
    importance = pd.DataFrame({"feature": FEATURES, "gain_importance": final_model.feature_importances_})
    importance.sort_values("gain_importance", ascending=False).round(4).to_csv(RESULTS / "feature_importance.csv", index=False)
    print("saved results/seed_r2.csv and results/feature_importance.csv")
