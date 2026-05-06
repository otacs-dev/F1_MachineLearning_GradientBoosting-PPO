import fastf1
import pandas as pd
import numpy as np

# Ativa o cache para não baixar os mesmos dados duas vezes
fastf1.Cache.enable_cache('Circuit_List') 

all_laps = []

# Defina aqui o intervalo de rodadas (ex: 1 a 16)
for rnd in range(1, 17):
    try:
        print(f"Baixando dados da Rodada {rnd}...")
        session = fastf1.get_session(2025, rnd, 'R')
        session.load(telemetry=False, weather=False, messages=False)

        # Extrai todas as voltas
        laps = session.laps.copy()

        # Adiciona informações de contexto do GP
        laps['GP'] = session.event['EventName']
        laps['Round'] = rnd

        # Seleciona apenas as colunas mais importantes para análise de dados
        cols = [
            'Round', 'GP', 'Driver', 'Team', 'LapNumber', 'Stint', 
            'PitOutTime', 'PitInTime', 'Compound', 'TyreLife',
            'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'IsAccurate'
        ]
        
        df_laps = laps[cols].copy()

        # CONVERSÃO: Transforma os objetos Timedelta em segundos (float)
        # Isso é o que permite fazer cálculos e gráficos depois
        time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
        for col in time_cols:
            df_laps[f'{col}Seconds'] = df_laps[col].dt.total_seconds()

        # Identifica se a volta foi um Pit Stop (ajuda na filtragem posterior)
        df_laps['IsPitStop'] = ~df_laps['PitInTime'].isna() | ~df_laps['PitOutTime'].isna()

        all_laps.append(df_laps)
        
    except Exception as e:
        print(f"Erro na rodada {rnd}: {e}")

# Consolida tudo em um único "Master DataFrame"
if all_laps:
    final_df = pd.concat(all_laps, ignore_index=True)

    # LIMPEZA FINAL: Remove colunas de Timedelta originais (não funcionam bem em CSV)
    final_df = final_df.drop(columns=['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'PitOutTime', 'PitInTime'])

    # Salva o arquivo mestre
    final_df.to_csv("F1_2025_Telemetry_Master.csv", index=False)
    print("\n Processamento concluído! Arquivo 'F1_2025_Telemetry_Master.csv' gerado.")