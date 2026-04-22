"""
Reduz ruído removendo outliers extremos por composto e GP.
Gera Dataset_Treino_F1_2025_DENOISED.csv
"""

import pandas as pd

INPUT = 'Dataset_Treino_F1_2025_FLAGS.csv'
OUTPUT = 'Dataset_Treino_F1_2025_DENOISED.csv'


def denoise(input_path=INPUT, output_path=OUTPUT):
    df = pd.read_csv(input_path)

    # Filtro robusto por IQR para reduzir ruído sem remover variação útil.
    def iqr_bounds(s):
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        return q1 - 2.5 * iqr, q3 + 2.5 * iqr

    mask_keep = pd.Series(True, index=df.index)

    groups = df.groupby(['Round', 'GP', 'Compound', 'Driver'])
    removed = 0
    for _, g in groups:
        if len(g) < 10:
            continue
        low, high = iqr_bounds(g['LapTimeSeconds'])
        drop_idx = g.index[(g['LapTimeSeconds'] < low) | (g['LapTimeSeconds'] > high)]
        if len(drop_idx) > 0:
            mask_keep.loc[drop_idx] = False
            removed += len(drop_idx)

    df_clean = df[mask_keep].copy().reset_index(drop=True)
    df_clean.to_csv(output_path, index=False)
    print(f'Removed {removed} extreme outliers; saved {output_path} ({len(df_clean)} rows)')
    return df_clean

if __name__ == '__main__':
    denoise()
