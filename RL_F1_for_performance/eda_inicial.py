import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Cria pasta output se não existir
os.makedirs('output/', exist_ok=True)


# 2. Carrega dataset
laps = pd.read_csv('f1_data/lap_data.csv')

laps['LapTime_sec'] = pd.to_timedelta(laps['LapTime']).dt.total_seconds()
compostos_validos = ["SOFT", "MEDIUM", "HARD"]
deg = laps[
    (laps["Compound"].isin(compostos_validos)) &
    (laps["TyreLife"] <= 40) &
    (laps["LapTime_sec"].between(75, 110))
].copy()

degradacao = (
    deg.groupby(["Compound", "TyreLife"])["LapTime_sec"]
    .mean()
    .reset_index()
)
 
plt.figure(figsize=(12, 5))
for comp, color in zip(compostos_validos, ["red", "gold", "gray"]):
    subset = degradacao[degradacao["Compound"] == comp]
    plt.plot(subset["TyreLife"], subset["LapTime_sec"], label=comp, color=color, linewidth=2)
 
plt.title("Análise de pace por Composto")
plt.xlabel("Idade do Pneu (voltas)")
plt.ylabel("Tempo Médio de Volta (s)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("output/", "pace_pneu.png"))
plt.close()
print("  ✅ Salvo: pace_pneu.png")