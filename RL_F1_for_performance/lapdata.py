import fastf1
import os
import pandas as pd

os.makedirs("fastf1_cache", exist_ok=True)  # cria a pasta se não existir
fastf1.Cache.enable_cache("fastf1_cache")  # cria pasta de cache local

all_laps = [] 

for rnd in range(1, 25):
    print(f"Carregando round {rnd}...")

    session = fastf1.get_session(2024, rnd, "R")  
    session.load(telemetry=False, weather=False)

    laps = session.laps[[
    "Driver", "LapNumber", "LapTime",
    "Position", "Compound", "TyreLife", "Stint", "Team"
    ]].copy()

    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    laps["Season"] = 2024
    laps["Round"] = rnd
    laps["RaceName"] = session.event["EventName"]


    all_laps.append(laps)

df = pd.concat(all_laps, ignore_index=True)
df.to_csv("lap_data.csv", index=False)