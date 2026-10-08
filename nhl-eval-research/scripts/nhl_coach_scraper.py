"""
Builds data/coach_tenures.csv: every head coach stint from 2021-22 to 2025-26.
Uses the public NHL API. For each team-season it checks the coach in the first and last game.
If they differ, it binary searches the schedule for the game where the coach changed.
"""

import asyncio
import time

import aiohttp
import pandas as pd

from config import ROOT

BASE = "https://api-web.nhle.com/v1"
SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026]
GAME_TYPE = 2  # regular season

TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "VAN", "VGK", "WPG", "WSH",
]

# Arizona became Utah in 2024-25
TEAMS_BY_SEASON = {
    20212022: TEAMS + ["ARI"],
    20222023: TEAMS + ["ARI"],
    20232024: TEAMS + ["ARI"],
    20242025: TEAMS + ["UTA"],
    20252026: TEAMS + ["UTA"],
}

MAX_CONCURRENT = 40
RETRY_LIMIT = 3
RETRY_DELAY = 1.0

CSV_FILE = ROOT / "data" / "coach_tenures.csv"

stats = {"requests": 0, "retries": 0, "errors": 0}


async def fetch_json(session, sem, url):
    for attempt in range(RETRY_LIMIT):
        try:
            async with sem:
                async with session.get(url) as resp:
                    stats["requests"] += 1
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status == 429:
                        stats["retries"] += 1
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    if resp.status == 404:
                        return None
                    resp.raise_for_status()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            stats["retries"] += 1
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    stats["errors"] += 1
    return None


async def get_team_schedule(session, sem, team, season):
    url = f"{BASE}/club-schedule-season/{team}/{season}"
    data = await fetch_json(session, sem, url)
    if not data:
        return []
    games = []
    for g in data.get("games", []):
        if g.get("gameType") != GAME_TYPE:
            continue
        games.append({
            "game_id": g["id"],
            "date": g["gameDate"],
            "home": g["homeTeam"]["abbrev"],
            "away": g["awayTeam"]["abbrev"],
        })
    return games


async def get_coach_for_game(session, sem, game_id):
    # returns (home coach, away coach, raw data)
    url = f"{BASE}/gamecenter/{game_id}/right-rail"
    data = await fetch_json(session, sem, url)
    if not data:
        return None, None, data

    gi = data.get("gameInfo", {})
    home_coach = None
    away_coach = None

    hc_obj = gi.get("homeTeam", {}).get("headCoach")
    if hc_obj and isinstance(hc_obj, dict):
        home_coach = hc_obj.get("default")
    elif hc_obj:
        home_coach = str(hc_obj)

    ac_obj = gi.get("awayTeam", {}).get("headCoach")
    if ac_obj and isinstance(ac_obj, dict):
        away_coach = ac_obj.get("default")
    elif ac_obj:
        away_coach = str(ac_obj)

    return home_coach, away_coach, data


async def verify_api(session, sem):
    url = f"{BASE}/club-schedule-season/TOR/20242025"
    sched = await fetch_json(session, sem, url)
    if not sched:
        print("could not fetch schedule")
        return False

    reg_games = [g for g in sched.get("games", []) if g.get("gameType") == GAME_TYPE]
    if not reg_games:
        print("no regular season games")
        return False

    gid = reg_games[0]["id"]
    hc, ac, _ = await get_coach_for_game(session, sem, gid)
    if hc and ac:
        return True

    print(f"game {gid} returned home={hc}, away={ac}")
    return False


async def find_change_point(session, sem, games, team, coach_start, coach_end):
    # first game index where the coach is no longer the opening-day coach
    lo, hi = 0, len(games) - 1
    change_idx = hi

    while lo <= hi:
        mid = (lo + hi) // 2
        gid = games[mid]["game_id"]
        hc, ac, _ = await get_coach_for_game(session, sem, gid)
        coach = hc if games[mid]["home"] == team else ac

        if coach == coach_start:
            lo = mid + 1
        else:
            change_idx = mid
            hi = mid - 1

    return change_idx


async def main():
    t0 = time.time()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=30)
    headers = {"User-Agent": "Mozilla/5.0"}

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:

        if not await verify_api(session, sem):
            print("API check failed")
            return

        # schedules for every team-season
        schedule_keys = []
        schedule_coros = []
        for season in SEASONS:
            for team in TEAMS_BY_SEASON[season]:
                schedule_keys.append((team, season))
                schedule_coros.append(get_team_schedule(session, sem, team, season))

        schedules = {}
        results = await asyncio.gather(*schedule_coros, return_exceptions=True)
        for key, result in zip(schedule_keys, results):
            if isinstance(result, Exception):
                print(f"error {key}: {result}")
                schedules[key] = []
            else:
                schedules[key] = sorted(result, key=lambda g: g["date"])

        # coach in the first and last game of each team-season
        coach_records = []
        change_candidates = []

        edge_keys = []
        edge_coros = []
        for (team, season), games in schedules.items():
            if not games:
                continue
            first_gid = games[0]["game_id"]
            last_gid = games[-1]["game_id"]
            edge_keys.append((team, season, "first"))
            edge_coros.append(get_coach_for_game(session, sem, first_gid))
            if first_gid != last_gid:
                edge_keys.append((team, season, "last"))
                edge_coros.append(get_coach_for_game(session, sem, last_gid))

        edge_results = await asyncio.gather(*edge_coros, return_exceptions=True)
        edge_map = {}
        for key, result in zip(edge_keys, edge_results):
            if isinstance(result, Exception):
                edge_map[key] = (None, None, None)
            else:
                edge_map[key] = result

        for (team, season), games in schedules.items():
            if not games:
                continue

            first_hc, first_ac, _ = edge_map.get((team, season, "first"), (None, None, None))
            first_game = games[0]
            coach_first = first_hc if first_game["home"] == team else first_ac

            if len(games) <= 1:
                if coach_first:
                    coach_records.append({
                        "coach": coach_first,
                        "team": team,
                        "season": season,
                        "start_date": first_game["date"],
                        "end_date": first_game["date"],
                        "games": 1,
                        "mid_season_hire": False,
                    })
                continue

            last_hc, last_ac, _ = edge_map.get((team, season, "last"), (None, None, None))
            last_game = games[-1]
            coach_last = last_hc if last_game["home"] == team else last_ac

            if coach_first and coach_last and coach_first == coach_last:
                coach_records.append({
                    "coach": coach_first,
                    "team": team,
                    "season": season,
                    "start_date": first_game["date"],
                    "end_date": last_game["date"],
                    "games": len(games),
                    "mid_season_hire": False,
                })
            elif coach_first and coach_last:
                change_candidates.append((team, season, games, coach_first, coach_last))
            else:
                print(f"could not read coaches for {team} {season} "
                      f"(first={coach_first}, last={coach_last})")

        # find the exact game of each mid-season change
        for team, season, games, coach_start, coach_end in change_candidates:
            change_idx = await find_change_point(session, sem, games, team, coach_start, coach_end)

            coach_records.append({
                "coach": coach_start,
                "team": team,
                "season": season,
                "start_date": games[0]["date"],
                "end_date": games[change_idx - 1]["date"] if change_idx > 0 else games[0]["date"],
                "games": change_idx,
                "mid_season_hire": False,
            })
            coach_records.append({
                "coach": coach_end,
                "team": team,
                "season": season,
                "start_date": games[change_idx]["date"],
                "end_date": games[-1]["date"],
                "games": len(games) - change_idx,
                "mid_season_hire": True,
            })

            # a second change in the same season is rare, just flag it
            mid_gid = games[len(games) // 2]["game_id"]
            mid_hc, mid_ac, _ = await get_coach_for_game(session, sem, mid_gid)
            mid_coach = mid_hc if games[len(games) // 2]["home"] == team else mid_ac
            if mid_coach and mid_coach != coach_start and mid_coach != coach_end:
                print(f"second change for {team} {season}, also had {mid_coach}")

    df = pd.DataFrame(coach_records)
    if df.empty:
        print("no data collected")
        return

    df = df.sort_values(["season", "team", "start_date"]).reset_index(drop=True)
    season_str = df["season"].astype(str)
    df["season_label"] = season_str.str[:4] + "-" + season_str.str[4:6]

    # the team each coach had right before this stint
    df["prior_team"] = ""
    for idx, row in df.iterrows():
        prior = df[
            (df["coach"] == row["coach"])
            & (df["end_date"] < row["start_date"])
        ].sort_values("end_date", ascending=False)
        if not prior.empty:
            df.at[idx, "prior_team"] = prior.iloc[0]["team"]

    col_order = [
        "coach", "team", "season_label", "start_date", "end_date",
        "games", "mid_season_hire", "prior_team", "season",
    ]
    df = df[col_order]
    df.to_csv(CSV_FILE, index=False)

    elapsed = time.time() - t0
    print(f"{len(df)} coach stints written to {CSV_FILE}")
    print(f"{stats['requests']} requests, {stats['retries']} retries, "
          f"{stats['errors']} errors, {elapsed:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
