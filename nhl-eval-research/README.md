# NHL forward offense research

This is a research project, not a finished tool. I wanted to know how much of a forward's offensive impact holds up when his situation changes, and how much of it depends on who he plays with and against. I started in April 2026 and paused at the end of August. This repo has the code, the results so far, and the things that didn't work.

## The question

When a forward changes coach or team, how well can we explain his offensive impact in the new season from how he plays, who his linemates are, and how he's used?

## Data

Five regular seasons, 2021-22 to 2025-26.

- **All Three Zones** (Corey Sznajder): hand-tracked microstats like zone entries, forecheck pressures and high danger passes. It's a paid subscription, so only a small sample is in `data/samples`.
- **Evolving Hockey**: xGAR, quality of competition, forward line combinations, forward and defense pairings, and player age. Also paid, so sample rows only.
- **Natural Stat Trick**: team and player Fenwick tables, zone starts, individual expected goals, and 5v5 on-ice rates. These are in `data/nst` in full.
- **NHL API**: head coaches by team and date, which gives the coach tenures in `data/coach_tenures.csv`. The scraper is `scripts/nhl_coach_scraper.py`.

## How it works

Each player-season is split by coach stint, because a mid-season coaching change means a different system.

I score five play styles for each player: shooter, playmaker, transition, forecheck and cycle, and net front. Each is an average of a few per-60 stats. Most of those stats are blended with his team's average, (raw + above team) / 2. That separates how the player plays from how his team plays, so the score reflects the player and not the system he's in. Three stats are used raw: offensive zone assists, low-to-high passes and retrievals leading to exits. Every stat is then z-scored within season and position, with players weighted by games tracked so small samples count less. The model only sees each player's average from earlier seasons.

The target is the average of two within-season z-scores. One is even strength offensive xGAR per game (the `xEVO` piece of Evolving Hockey's xGAR). The other is his on-ice xGF/60 at 5v5, blended with his team's number the same way.

The features are:

- the player's five style scores, and the same five for his top two linemates
- three team style numbers: possession, shot efficiency, and forecheck-or-cycle versus rush
- last season's target
- last season's ice time per game
- age
- quality of competition, which is the average strength of his opponents
- zone start rates
- how many earlier seasons he and each linemate were already on the top line together
- the average style of his top six defense partners, weighted by time together

The model is XGBoost with depth 3 and heavy regularization. It's trained on 549 forward seasons where the coach or team changed (2022-23 to 2025-26, since 2021-22 has no earlier season to compare to). The score is R squared from 5-fold cross-validation, repeated over 20 random seeds.

## Results

The 20-seed mean R squared is **0.462** (std 0.011, range 0.440 to 0.479). Using last season's number alone gets about 0.32 under the same folds.

Because those folds are random by row, the same player can appear in both train and test in different seasons. So I added two stricter checks (`scripts/validation_checks.py`):

| check | model R squared | last season alone |
|---|---|---|
| random rows (the main result) | 0.462 | 0.318 |
| folds grouped by player | 0.452 | 0.309 |
| train 2022-23 to 2023-24, test 2024-25 (174 rows) | 0.394 | 0.235 |
| train 2022-23 to 2024-25, test 2025-26 (95 rows) | 0.396 | 0.152 |

The model drops about 0.07 when it has to predict a later season it hasn't seen, but it still beats the baseline by a wide margin. The test sets are small, so I wouldn't read much into the difference between the last two rows.

The biggest single gain along the way came from adding linemates and quality of competition, about +0.12 on the early version, positive on all 20 seeds. By feature importance the model leans most on last season's target, quality of competition, offensive zone start rate, and the player's own transition and shooter scores (`results/feature_importance.csv`, one fit, so treat it as rough).

Every step I tried, with the score and whether I kept it, is in `results/results.csv`. The earlier numbers there were measured on the feature set at the time, so the rows don't add up exactly. The per-seed scores are in `results/seed_r2.csv`.

## What didn't work

- NHL Edge skating speed: very repeatable year to year, but it added nothing.
- Height and weight: nothing.
- Speed trend over time: worse, mostly because of missing data.
- Pace of play: nothing.
- Splitting shot volume from shot quality: worse than averaging them.
- Stats with almost no year to year stability, like carries with chances. I dropped them.
- An earlier version used team system stats to predict RAPM. They added nothing beyond a player's own RAPM history, so I rebuilt the model.
- Predicting breakouts and busts: team style made assist predictions worse for players who changed teams, and the biggest swings stayed unpredictable.

## Limits

- It doesn't forecast. Linemates, quality of competition and zone starts come from the same season as the target, so you can't run this before the season. It describes what goes with offensive impact.
- For traded players, only the stint with the most games is used.
- All Three Zones tracks about 400 games a season, so low-usage players are noisy. Stints under 10 tracked games are dropped.
- Forwards only, and only players whose coach or team changed.
- Changes under about 0.015 R squared are inside the noise between seeds. Keeping the defense partner feature was a judgment call.

## Running it

```
pip install -r requirements.txt
cd scripts
python nhl_coach_scraper.py      # optional, the output is already in data/
python assign_coach_tenures.py
python player_archetypes.py
python xg_clean.py
python model.py
python validation_checks.py
```

The paid data isn't included, so this only runs end to end if you have your own copies. Put them in `data/private/` with these names:

```
a3z_players_21_26.xlsx            (Game Stats sheet)
eh_xevo_xevd.csv
eh_rapm_ev.csv
eh_quality_of_competition.csv
eh_forward_lines/21-22.csv ... 25-26.csv
eh_forwards_and_d/21-22.csv ... 25-26.csv
```

The files in `data/samples` show the columns and format of each one (25 random rows each). The All Three Zones sample is taken after the coach assignment step, so it has two extra columns, `coach` and `coach_stint`.

## Credit

Corey Sznajder and All Three Zones, Evolving Hockey, Natural Stat Trick, and NHL.com.
