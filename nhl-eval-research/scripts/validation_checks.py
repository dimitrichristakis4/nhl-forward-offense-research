import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from config import RESULTS
from model import model_df, FEATURES, TARGET, MODEL_PARAMS, r2_score_simple

# The main model uses random 5-fold CV on player-seasons, so the same player can be in
# both train and test (different seasons). These two checks remove that overlap.
#   player_grouped: every player's rows stay together in one fold
#   walk_forward: train on earlier seasons only, test on one later season
# The baseline predicts the target from last season's value alone (linear fit on the train rows).

N_SEEDS = 20

def fit_and_score(train, test, seed):
    model = XGBRegressor(random_state=seed, **MODEL_PARAMS)
    model.fit(train[FEATURES].values, train[TARGET].values)
    r2_model = r2_score_simple(test[TARGET].values, model.predict(test[FEATURES].values))

    base = LinearRegression().fit(train[["blend_prev"]].values, train[TARGET].values)
    r2_base = r2_score_simple(test[TARGET].values, base.predict(test[["blend_prev"]].values))
    return r2_model, r2_base

def summarize(name, split, rows, n_train, n_test):
    r2m = np.array([r[0] for r in rows])
    r2b = np.array([r[1] for r in rows])
    return {
        "check": name, "split": split, "n_train": n_train, "n_test": n_test,
        "model_r2_mean": round(r2m.mean(), 4), "model_r2_std": round(r2m.std(), 4),
        "baseline_r2_mean": round(r2b.mean(), 4), "baseline_r2_std": round(r2b.std(), 4),
    }

df = model_df.reset_index(drop=True)
out = []

# random rows, same as the main result
rows, n_tr, n_te = [], [], []
for seed in range(N_SEEDS):
    fold_scores = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(df):
        fold_scores.append(fit_and_score(df.iloc[tr], df.iloc[te], seed))
        n_tr.append(len(tr)); n_te.append(len(te))
    rows.append(tuple(np.mean(fold_scores, axis=0)))
out.append(summarize("random_rows_5fold", "all seasons", rows, int(np.mean(n_tr)), int(np.mean(n_te))))

# players split into 5 groups at random, new grouping for each seed
players = df["player_key"].unique()
rows, n_tr, n_te = [], [], []
for seed in range(N_SEEDS):
    rng = np.random.RandomState(seed)
    group_of = dict(zip(players, rng.permutation(len(players)) % 5))
    folds = df["player_key"].map(group_of).values
    fold_scores = []
    for k in range(5):
        tr, te = df[folds != k], df[folds == k]
        fold_scores.append(fit_and_score(tr, te, seed))
        n_tr.append(len(tr)); n_te.append(len(te))
    rows.append(tuple(np.mean(fold_scores, axis=0)))
out.append(summarize("player_grouped_5fold", "all seasons", rows, int(np.mean(n_tr)), int(np.mean(n_te))))

# train on every earlier season, test on one season
seasons = sorted(df["season2"].unique())
for test_season in seasons[2:]:
    tr = df[df["season2"] < test_season]
    te = df[df["season2"] == test_season]
    rows = [fit_and_score(tr, te, seed) for seed in range(N_SEEDS)]
    out.append(summarize("walk_forward", f"test {test_season}", rows, len(tr), len(te)))

result = pd.DataFrame(out)
print(result.to_string(index=False))
result.to_csv(RESULTS / "validation_checks.csv", index=False)
