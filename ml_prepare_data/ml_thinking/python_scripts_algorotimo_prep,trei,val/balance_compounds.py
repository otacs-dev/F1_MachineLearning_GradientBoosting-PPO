"""
Balanceamento de Compostos
--------------------------
Lê Dataset_Treino_F1_2025_CLEANED.csv e faz undersampling dos compostos
mais abundantes (HARD, MEDIUM) para equilibrar em relação ao SOFT.
Gera Dataset_Treino_F1_2025_BALANCED.csv
"""

import pandas as pd

INPUT = 'Dataset_Treino_F1_2025_CLEANED.csv'
OUTPUT = 'Dataset_Treino_F1_2025_BALANCED.csv'


def _sample_group(gdf, n, seed):
    if len(gdf) <= n:
        return gdf
    return gdf.sample(n=n, random_state=seed)


def balance_compounds(input_path=INPUT, output_path=OUTPUT, target_frac=0.15, seed=42):
    df = pd.read_csv(input_path)
    counts = df['Compound'].value_counts()
    print('Compostos atuais:', counts.to_dict())

    # Alvo por composto: manter pelo menos target_frac do total ou o tamanho do SOFT.
    total = len(df)
    soft_count = counts.get('SOFT', 0)
    target = max(int(total * target_frac), soft_count)
    print(f'Target por composto: {target}')

    parts = []
    for comp, cnt in counts.items():
        comp_df = df[df['Compound'] == comp].copy()
        if cnt <= target:
            parts.append(comp_df)
        else:
            if 'is_wet_session' in comp_df.columns:
                sampled_parts = []
                for wet_flag, frac in comp_df['is_wet_session'].value_counts(normalize=True).items():
                    local_target = max(1, int(round(target * frac)))
                    stratum = comp_df[comp_df['is_wet_session'] == wet_flag]
                    sampled_parts.append(_sample_group(stratum, local_target, seed))
                sampled = pd.concat(sampled_parts, ignore_index=True)
                sampled = _sample_group(sampled, target, seed)
            else:
                sampled = _sample_group(comp_df, target, seed)
            parts.append(sampled)
            print(f'Undersampled {comp}: {cnt} -> {len(sampled)}')

    df_bal = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    print('Balanced counts:', df_bal['Compound'].value_counts().to_dict())
    df_bal.to_csv(output_path, index=False)
    print(f'Arquivo salvo: {output_path} ({len(df_bal)} linhas)')
    return df_bal

if __name__ == '__main__':
    balance_compounds()
