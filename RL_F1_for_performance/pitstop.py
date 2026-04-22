import fastf1
import os
import pandas as pd

os.makedirs("fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache("fastf1_cache")

all_pits = []

for rnd in range(1, 25):
    print(f"Carregando round {rnd}...")

    session = fastf1.get_session(2024, rnd, "R")
    session.load(telemetry=False, weather=False)

    laps = session.laps.copy()

    # filtra só voltas onde o carro entrou no pit
    pit_laps = laps[laps["PitInTime"].notna()].copy()
    pit_laps["PitDuration_s"] = (
        pit_laps["PitOutTime"] - pit_laps["PitInTime"]
    ).dt.total_seconds()

    pit_laps["Season"]   = 2024
    pit_laps["Round"]    = rnd
    pit_laps["RaceName"] = session.event["EventName"]

    all_pits.append(pit_laps[[
        "Season", "Round", "RaceName",
        "Driver", "Team", "LapNumber", "Stint",
        "Compound", "PitDuration_s"
    ]])
df = pd.concat(all_pits, ignore_index=True)
df.to_csv("pit_stops.csv", index=False)
print(f"Salvo! {len(df)} linhas.")