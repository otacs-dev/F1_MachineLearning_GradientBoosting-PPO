import fastf1
import os
import pandas as pd

os.makedirs("fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache("fastf1_cache")

all_events = []
for rnd in range(1, 25):
    print(f"Carregando round {rnd}...")

    session = fastf1.get_session(2024, rnd, "R")
    session.load(telemetry=False, weather=False, messages=True)

    track_status = session.track_status.copy()
    status_map = {
        "1": "AllClear",
        "2": "Yellow",
        "4": "SafetyCar",
        "5": "RedFlag",
        "6": "VSC_deployed",
        "7": "VSC_ending",
    }

    track_status["StatusName"] = track_status["Status"].map(status_map).fillna("Unknown")
    track_status["Season"]     = 2024
    track_status["Round"]      = rnd
    track_status["RaceName"]   = session.event["EventName"]

    all_events.append(track_status[[
        "Season", "Round", "RaceName", "Status", "StatusName", "Time"
    ]])
df = pd.concat(all_events, ignore_index=True)
df.to_csv("race_events.csv", index=False)
print(f"Salvo! {len(df)} linhas.")