import fastf1
import pandas as pd

# O loop para as rodadas que você deseja
for rnd in range(1, 17): # 1 a 16
    try:
        session = fastf1.get_session(2025, rnd, 'R')
        session.load(telemetry=False, weather=False, messages=False)

        # Extraindo os resultados
        drivers = session.results[["FullName", "TeamName", "ClassifiedPosition"]]
        
        # Nome do arquivo usando aspas simples dentro da f-string
        filename = f"Round_{rnd}_{session.event['EventName']}_2025_results.csv"
        
        # Salvando
        drivers.to_csv(filename, index=False)
        print(f"Sucesso: {filename} salvo.")
        
    except Exception as e:
        print(f"Erro na rodada {rnd}: {e}")

        