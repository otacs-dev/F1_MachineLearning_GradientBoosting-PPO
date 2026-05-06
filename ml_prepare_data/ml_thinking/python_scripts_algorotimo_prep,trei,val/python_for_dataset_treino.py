# Script: merge_telemetry_drivers.py
# Objetivo: Mesclar dados de telemetria com informações de pilotos + limpeza

import pandas as pd
import os

def merge_telemetry_drivers(
    telemetry_path='Circuit_List/F1_2025_Telemetry_Master.csv',
    drivers_path='Driver_List/Pilotos/Pilotos_2025.csv',
    output_path='Dataset_Treino_F1_2025.csv'
):
    """
    Mescla dados de telemetria com informações de pilotos.
    
    Parâmetros:
    -----------
    telemetry_path : str
        Caminho para o CSV de telemetria
    drivers_path : str
        Caminho para o CSV de pilotos
    output_path : str
        Caminho para salvar o dataset final
    """
    
    print("=" * 60)
    print("MERGE: Telemetria + Informações de Pilotos")
    print("=" * 60)
    
    # 1. Carrega os dados
    print("\n📂 Carregando dados...")
    df_telemetria = pd.read_csv(telemetry_path)
    df_pilotos = pd.read_csv(drivers_path)
    
    print(f"   ✓ Telemetria: {len(df_telemetria)} linhas")
    print(f"   ✓ Pilotos: {len(df_pilotos)} linhas")
    
    # 2. Valida se o LapTimeSeconds já existe (se não, converte)
    if 'LapTimeSeconds' not in df_telemetria.columns:
        print("\n⚠️  'LapTimeSeconds' não encontrado, convertendo de 'LapTime'...")
        df_telemetria['LapTimeSeconds'] = pd.to_timedelta(df_telemetria['LapTime']).dt.total_seconds()
    
    # 3. MERGE: Associa piloto pelo identificador comum (Driver / Abbreviation)
    print("\n🔗 Mesclando datasets...")
    df_merged = pd.merge(
        df_telemetria,
        df_pilotos,
        left_on='Driver',        # Coluna na telemetria (ex: 'VER')
        right_on='Abbreviation', # Coluna no cadastro de pilotos
        how='left'               # Keep all telemetry rows, even if no driver match
    )
    
    print(f"   ✓ Dataset mesclado: {len(df_merged)} linhas")
    
    # 4. Verifica se houve matches
    unmatched = df_merged['FullName'].isna().sum()
    if unmatched > 0:
        print(f"   ⚠️  {unmatched} linhas sem match de piloto (pilotos não encontrados)")
    
    # 5. Reorganiza colunas: prioriza dados importantes no início
    col_order = [
        # Informações da corrida
        'Round', 'GP',
        # Informações do piloto (do merge)
        'Driver', 'Abbreviation', 'FullName', 'Number', 'TeamName', 'TeamColor',
        # Dados da volta
        'LapNumber', 'Stint', 'Compound', 'TyreLife',
        # Tempos (essencial para IA)
        'LapTimeSeconds', 'Sector1TimeSeconds', 'Sector2TimeSeconds', 'Sector3TimeSeconds',
        # Metadados
        'IsAccurate', 'IsPitStop', 'Team'
    ]
    
    # Filtra apenas colunas que existem
    col_order = [col for col in col_order if col in df_merged.columns]
    df_final = df_merged[col_order]
    
    # 6. Limpeza opcional: remove linhas sem LapTimeSeconds válido (voltas parciais)
    print("\n🧹 Limpando dados...")
    valid_before = len(df_final)
    df_final = df_final[df_final['LapTimeSeconds'].notna()]
    valid_after = len(df_final)
    removed = valid_before - valid_after
    print(f"   ✓ Removidas {removed} linhas com LapTimeSeconds inválido")
    
    # 7. Salva o dataset final
    print("\n💾 Salvando dataset final...")
    df_final.to_csv(output_path, index=False)
    print(f"   ✓ Arquivo salvo: {output_path}")
    
    # 8. Estatísticas finais
    print("\n📊 Estatísticas do Dataset Final:")
    print(f"   - Total de linhas: {len(df_final)}")
    print(f"   - Total de colunas: {len(df_final.columns)}")
    print(f"   - Corridas (GPs): {df_final['GP'].nunique()}")
    print(f"   - Pilotos únicos: {df_final['FullName'].nunique()}")
    print(f"   - Tempo médio de volta: {df_final['LapTimeSeconds'].mean():.2f}s")
    
    # 9. Amostra visual
    print("\n👀 Amostra do dataset final:")
    print(df_final[['FullName', 'Number', 'LapTimeSeconds', 'Compound', 'GP']].head(10).to_string())
    
    return df_final

if __name__ == "__main__":
    df_final = merge_telemetry_drivers()
    print("\n✅ Merge concluído com sucesso!")