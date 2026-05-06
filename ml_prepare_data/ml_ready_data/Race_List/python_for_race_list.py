import fastf1
import pandas as pd
import os
from pathlib import Path

# Configuração de Cache
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / 'f1_cache_race_list'
WEATHER_DIR = SCRIPT_DIR / 'Clima'

if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

def get_weather_data():
    print("🌤️ Iniciando coleta de dados climáticos de 2025...")
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    
    weather_master = []

    # Percorrendo as rodadas
    for round_num in range(1, 18): # Ajustado para cobrir até a rodada 17 se necessário
        try:
            print(f"Buscando clima: Round {round_num}...", end="\r")
            session = fastf1.get_session(2025, round_num, 'R')
            
            # Carregamos apenas o clima. laps=False economiza muita memória e tempo.
            session.load(telemetry=False, weather=True, laps=False)
            
            # Acessamos os dados climáticos brutos diretamente da sessão
            weather_df = session.weather_data.copy()
            
            if not weather_df.empty:
                weather_df['RoundNumber'] = round_num
                weather_df['EventName'] = session.event['EventName']
                weather_master.append(weather_df)
            else:
                print(f"\n⚠️ Sem dados climáticos para o Round {round_num}")
            
        except Exception as e:
            print(f"\n❌ Erro no Round {round_num}: {e}")

    if weather_master:
        final_weather_df = pd.concat(weather_master, ignore_index=True)
        
        # Salvando o Master de Clima
        file_path = WEATHER_DIR / 'Weather_2025_Master.csv'
        final_weather_df.to_csv(file_path, index=False)
        
        print(f"\n\n✅ Sucesso! Arquivo '{file_path}' gerado.")
        print(f"📊 Total de registros: {len(final_weather_df)}")
        print(final_weather_df[['EventName', 'AirTemp', 'TrackTemp', 'Rainfall']].head())
    else:
        print("\n\n⚠️ Falha crítica: Nenhum dado foi consolidado.")

if __name__ == "__main__":
    get_weather_data()