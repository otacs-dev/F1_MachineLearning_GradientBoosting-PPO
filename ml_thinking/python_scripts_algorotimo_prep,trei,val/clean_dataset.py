"""
Data Cleaning Script - F1 2025 Telemetry Dataset
================================================

Objetivo: Remover dados inconsistentes, pit stops, e valores nulos
que atrapalham o treinamento da IA.

Resultado: Dataset_Treino_F1_2025_CLEANED.csv (pronto para ML)
"""

import pandas as pd
import numpy as np
from pathlib import Path

WEATHER_CANDIDATES = [
    'Race_List/Clima/Weather_2025_Master.csv',
    'Clima/Weather_2025_Master.csv',
]


def _resolve_weather_path(weather_path=None):
    if weather_path:
        candidate = Path(weather_path)
        return str(candidate) if candidate.exists() else None
    for candidate in WEATHER_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _merge_weather_features(df, weather_path=None):
    resolved_path = _resolve_weather_path(weather_path=weather_path)
    if resolved_path is None:
        print(f"\n   [Weather] Arquivo não encontrado em {WEATHER_CANDIDATES} (seguindo sem clima)")
        return df

    weather = pd.read_csv(resolved_path)
    required = {'RoundNumber', 'EventName', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'Rainfall'}
    if not required.issubset(set(weather.columns)):
        print("\n   [Weather] Colunas esperadas não encontradas (seguindo sem clima)")
        return df

    weather['Rainfall'] = weather['Rainfall'].astype(str).str.lower().map({'true': True, 'false': False})
    weather['Rainfall'] = weather['Rainfall'].fillna(False).astype(bool)

    agg = (
        weather.groupby(['RoundNumber', 'EventName'], as_index=False)
        .agg(
            air_temp_mean=('AirTemp', 'mean'),
            air_temp_std=('AirTemp', 'std'),
            track_temp_mean=('TrackTemp', 'mean'),
            track_temp_std=('TrackTemp', 'std'),
            humidity_mean=('Humidity', 'mean'),
            pressure_mean=('Pressure', 'mean'),
            wind_speed_mean=('WindSpeed', 'mean'),
            rainfall_ratio=('Rainfall', 'mean'),
            had_rain=('Rainfall', 'max'),
        )
    )
    agg.rename(columns={'RoundNumber': 'Round', 'EventName': 'GP'}, inplace=True)

    merged = df.merge(agg, on=['Round', 'GP'], how='left')
    merged['had_rain'] = merged['had_rain'].fillna(False).astype(bool)
    merged['rainfall_ratio'] = merged['rainfall_ratio'].fillna(0.0)
    merged['is_wet_session'] = merged['had_rain'] | (merged['rainfall_ratio'] >= 0.10)
    merged['temp_delta_track_air'] = merged['track_temp_mean'] - merged['air_temp_mean']
    print(f"\n   [Weather] Fonte: {resolved_path}")
    print(f"   [Weather] Clima anexado em {merged['track_temp_mean'].notna().mean()*100:.1f}% das linhas")
    return merged

def clean_dataset(
    input_path='Dataset_Treino_F1_2025.csv',
    output_path='Dataset_Treino_F1_2025_CLEANED.csv',
    weather_path=None,
):
    """
    Limpa o dataset de telemetria F1 2025 para treinamento de IA.
    
    Parâmetros:
    -----------
    input_path : str
        Caminho do CSV original
    output_path : str
        Caminho do CSV limpo
    """
    
    print("\n" + "="*70)
    print("🧹 LIMPEZA DE DATASET - F1 2025 TELEMETRY")
    print("="*70)
    
    # ── 1. CARREGA DADOS ──────────────────────────────────────────────────
    print("\n📂 Carregando dataset original...")
    df = pd.read_csv(input_path)
    rows_original = len(df)
    print(f"   ✓ Carregado: {rows_original:,} linhas × {len(df.columns)} colunas")
    
    # ── 2. ESTATÍSTICAS ANTES DA LIMPEZA ──────────────────────────────────
    print("\n📊 Análise Antes da Limpeza:")
    print(f"   - Pit stops (IsPitStop=True): {(df['IsPitStop']==True).sum():,}")
    print(f"   - LapTimeSeconds nulo: {df['LapTimeSeconds'].isna().sum():,}")
    print(f"   - IsAccurate=False: {(df['IsAccurate']==False).sum():,}")
    print(f"   - Team diferente de TeamName: {(df['Team'] != df['TeamName']).sum():,}")
    
    # ── 3. FILTRO 1: REMOVE PIT STOPS ─────────────────────────────────────
    print("\n🔧 Aplicando Filtros...")
    print("\n   [1/4] Removendo pit stops (IsPitStop==True)...")
    df_before = len(df)
    df = df[df['IsPitStop'] == False]
    pit_removed = df_before - len(df)
    print(f"        ✓ Removidas {pit_removed:,} linhas ({pit_removed/df_before*100:.1f}%)")
    
    # ── 4. FILTRO 2: REMOVE VALORES NULOS ─────────────────────────────────
    print("\n   [2/4] Removendo LapTimeSeconds nulo...")
    df_before = len(df)
    df = df[df['LapTimeSeconds'].notna()]
    nulos_removed = df_before - len(df)
    print(f"        ✓ Removidas {nulos_removed:,} linhas ({nulos_removed/df_before*100:.1f}%)")
    
    # ── 5. FILTRO 3: REMOVE DADOS COM BAIXA ACURÁCIA ──────────────────────
    print("\n   [3/4] Removendo dados com IsAccurate==False...")
    df_before = len(df)
    df = df[df['IsAccurate'] != False]
    inaccurate_removed = df_before - len(df)
    print(f"        ✓ Removidas {inaccurate_removed:,} linhas ({inaccurate_removed/df_before*100:.1f}%)")
    
    # ── 6. FILTRO 4: VALIDAÇÃO DE SECTOR TIMES ────────────────────────────
    print("\n   [4/4] Validando integridade de sector times...")
    # Calcula a soma dos 3 setores
    mask_valid_sectors = (
        df['Sector1TimeSeconds'].notna() & 
        df['Sector2TimeSeconds'].notna() & 
        df['Sector3TimeSeconds'].notna()
    )
    df_before = len(df)
    
    # Keep only rows where sectors exist and sum approximately to lap time
    # (com tolerância de 2 segundos para margem de erro)
    df_with_sectors = df[mask_valid_sectors].copy()
    df_with_sectors['sector_sum'] = (
        df_with_sectors['Sector1TimeSeconds'] + 
        df_with_sectors['Sector2TimeSeconds'] + 
        df_with_sectors['Sector3TimeSeconds']
    )
    
    # Keep rows where sector sum é razoavelmente próximo ao lap time
    df_valid = df_with_sectors[
        abs(df_with_sectors['sector_sum'] - df_with_sectors['LapTimeSeconds']) <= 2.0
    ].copy()
    df_valid.drop('sector_sum', axis=1, inplace=True)
    
    # Rows sem sectors completos (é ok, apenas faltam dados)
    df_incomplete = df[~mask_valid_sectors]
    
    df = pd.concat([df_valid, df_incomplete], ignore_index=True)
    sectors_removed = df_before - len(df)
    print(f"        ✓ Removidas {sectors_removed:,} linhas ({sectors_removed/df_before*100:.1f}%)")
    
    # ── 7. LIMPEZA ADICIONAL (opcional) ───────────────────────────────────
    print("\n   [Extra] Removendo duplicatas por (Round, Driver, LapNumber, Stint)...")
    df_before = len(df)
    df = df.drop_duplicates(
        subset=['Round', 'Driver', 'LapNumber', 'Stint'],
        keep='first'
    )
    dupes_removed = df_before - len(df)
    if dupes_removed > 0:
        print(f"        ✓ Removidas {dupes_removed:,} duplicatas")
    else:
        print(f"        ✓ Nenhuma duplicata encontrada")
    
    # ── 8. TRATAMENTO DE VALORES FALTANTES MENORES ────────────────────────
    print("\n   [Extra] Tratando valores faltantes em FullName/Number...")
    # Fill missing FullName with Driver abbreviation if not found
    df.loc[df['FullName'].isna(), 'FullName'] = df.loc[df['FullName'].isna(), 'Driver']
    
    # Fill missing Number with placeholder
    df.loc[df['Number'].isna(), 'Number'] = 999.0
    
    print(f"        ✓ Preenchidas {df['FullName'].isna().sum()} células de FullName")
    print(f"        ✓ Preenchidas {(df['Number']==999).sum()} células de Number")

    print("\n   [Extra] Tratando Compound faltante...")
    missing_compound_before = int(df['Compound'].isna().sum())
    if missing_compound_before > 0:
        # 1) tenta preencher com o composto modal do mesmo piloto
        mode_by_driver = (
            df[df['Compound'].notna()]
            .groupby('Driver')['Compound']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        )
        df.loc[df['Compound'].isna(), 'Compound'] = (
            df.loc[df['Compound'].isna(), 'Driver'].map(mode_by_driver)
        )

        # 2) fallback global para garantir zero nulos
        if df['Compound'].isna().any():
            global_mode = df['Compound'].dropna().mode()
            fallback = global_mode.iloc[0] if not global_mode.empty else 'UNKNOWN'
            df['Compound'] = df['Compound'].fillna(fallback)

    missing_compound_after = int(df['Compound'].isna().sum())
    print(f"        ✓ Compound nulo antes: {missing_compound_before}")
    print(f"        ✓ Compound nulo depois: {missing_compound_after}")
    
    # ── 9. REORDENAÇÃO E RESET ────────────────────────────────────────────
    print("\n   [Extra] Reordenando dataset por Round, GP, Driver, LapNumber...")
    df = df.sort_values(['Round', 'GP', 'Driver', 'LapNumber']).reset_index(drop=True)
    print(f"        ✓ Dataset reordenado")
    
    # ── 10. ENRIQUECIMENTO COM CLIMA ──────────────────────────────────────
    df = _merge_weather_features(df, weather_path=weather_path)

    # ── 11. ESTATÍSTICAS FINAIS ───────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 RESUMO DA LIMPEZA")
    print("="*70)
    
    rows_removed = rows_original - len(df)
    rows_kept = len(df)
    
    print(f"\n📈 Resumo Quantitativo:")
    print(f"   Linhas originais:  {rows_original:>10,}")
    print(f"   Linhas removidas:  {rows_removed:>10,} ({rows_removed/rows_original*100:>5.1f}%)")
    print(f"   Linhas finais:     {rows_kept:>10,} ({rows_kept/rows_original*100:>5.1f}%)")
    
    print(f"\n🏁 Dataset Limpo - Estatísticas Finais:")
    print(f"   - Corridas (Rounds): {df['Round'].nunique()}/16")
    print(f"   - Pilotos únicos: {df['Driver'].nunique()}")
    print(f"   - Times: {df['TeamName'].nunique()}")
    compounds = [str(c) for c in df['Compound'].dropna().unique()]
    print(f"   - Compostos: {len(compounds)} → {', '.join(compounds)}")
    
    lap_stats = df['LapTimeSeconds'].describe()
    print(f"\n⏱️  Tempo de Volta (LapTimeSeconds):")
    print(f"   Min:    {lap_stats['min']:>8.2f}s")
    print(f"   Max:    {lap_stats['max']:>8.2f}s")
    print(f"   Media:  {lap_stats['mean']:>8.2f}s")
    print(f"   Mediana:{lap_stats['50%']:>8.2f}s")
    print(f"   StdDev: {lap_stats['std']:>8.2f}s")
    
    print(f"\n💾 Distribuição por Composto:")
    compound_dist = df['Compound'].value_counts().sort_index()
    for comp, count in compound_dist.items():
        print(f"   {comp:>12}: {count:>6,} ({count/len(df)*100:>5.1f}%)")
    
    print(f"\n🔗 Distribuição por Piloto (top 10):")
    driver_dist = df['Driver'].value_counts().head(10)
    for driver, count in driver_dist.items():
        pil_name = df[df['Driver']==driver]['FullName'].iloc[0]
        print(f"   {driver} ({pil_name[:20]:<20}): {count:>5,} voltas")
    
    # ── 12. SALVA DATASET LIMPO ───────────────────────────────────────────
    print(f"\n💾 Salvando dataset limpo...")
    df.to_csv(output_path, index=False)
    print(f"   ✓ Arquivo salvo: {output_path}")
    print(f"   ✓ Tamanho: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    # ── 13. COMPARAÇÃO COM AMBIENTE ───────────────────────────────────────
    print(f"\n" + "="*70)
    print("✅ DATASET PRONTO PARA ML TRAINING")
    print("="*70)
    print(f"\n📌 Próximos Passos:")
    print(f"   1. Usar {output_path} com f1_env.py para treinar IA")
    print(f"   2. Dividir em train/val/test com stratificação por piloto")
    print(f"   3. Normalizar features para entrada da rede neural")
    print(f"   4. Treinar modelo de regressão ou classificação")
    
    print(f"\n🚀 Pronto para começar!\n")
    
    return df

if __name__ == "__main__":
    # Executa a limpeza
    df_clean = clean_dataset(
        input_path='Dataset_Treino_F1_2025.csv',
        output_path='Dataset_Treino_F1_2025_CLEANED.csv'
    )
