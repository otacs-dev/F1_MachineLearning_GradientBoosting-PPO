import fastf1
import os
import pandas as pd

os.makedirs("fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache("fastf1_cache")

all_laps = []
for rnd in range(1, 25):
    print(f"Carregando round {rnd}...")

    session = fastf1.get_session(2024, rnd, "R")
    session.load(telemetry=False, weather=False)

    laps = session.laps.copy()
    laps["Round"] = rnd
    laps["RaceName"] = session.event["EventName"]

    all_laps.append(laps)

df = pd.concat(all_laps, ignore_index=True)
df["LapTime_s"] = df["LapTime"].dt.total_seconds()

# filtra só voltas limpas e precisas
df = df[df["IsAccurate"] == True]
df = df[df["Round"] != 9]  # exclui GP do Canadá por ritmo discrepante da Ferrari

PILOTOS_REFERENCIA = [
    "VER",                # Red Bull — só Verstappen 
    "NOR", "PIA",         # McLaren — Norris e Piastri
    "LEC", "SAI",         # Ferrari — Leclerc e Sainz
    "HAM", "RUS",         # Mercedes — Hamilton e Russell
    "ALO", "STR",         # Aston Martin — Alonso e Stroll
    "HUL", "MAG", "BEA",  # Haas — Hulkenberg, Magnussen e Bearman (só nos GPs do Azerbaijão e São Paulo)
    "GAS", "OCO",         # Alpine — Gasly e Ocon
    "TSU", "RIC", "LAW",  # Racing Bulls — Tsunoda, Ricciardo e Lawson (A partir do GP dos EUA)
    "ALB", "SAR", "COL",  # Williams — Albon, Sargeant e Colapinto (A partir do GP da Itália)
    "BOT", "ZHO",         # Sauber — Bottas e Zhou
]
df = df[df["Driver"].isin(PILOTOS_REFERENCIA)]

# calcula tempo médio de volta por equipe
team_perf = (
    df.groupby("Team")["LapTime_s"]
    .mean()
    .reset_index()
    .rename(columns={"LapTime_s": "AvgLapTime_s"})
)

# ranking: menor tempo = melhor = rank 1
team_perf["PerfRank"] = team_perf["AvgLapTime_s"].rank(ascending=True).astype(int)
team_perf = team_perf.sort_values("PerfRank")
team_perf["GapToFirst_s"] = team_perf["AvgLapTime_s"] - team_perf["AvgLapTime_s"].min()
team_perf["Gap"] = team_perf["GapToFirst_s"].apply(lambda x: f"+{x:.3f}s" if x > 0 else "REF")


team_perf.to_csv("team_performances.csv", index=False)
print(team_perf[["Team", "AvgLapTime_s", "Gap", "PerfRank"]])

