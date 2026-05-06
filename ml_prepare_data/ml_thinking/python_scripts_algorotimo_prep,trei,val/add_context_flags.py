"""
Adiciona flags de contexto ao dataset limpo:
- first_lap (LapNumber == 1)
- first_stint_lap (TyreLife == 1)
- prev_was_pit (previous row for same driver had IsPitStop True)
- is_outlier (z-score per driver)
- sc_lap (heurística: lap-level median time jump)

Gera Dataset_Treino_F1_2025_FLAGS.csv
"""

import pandas as pd
import numpy as np

INPUT = 'Dataset_Treino_F1_2025_CLEANED.csv'
OUTPUT = 'Dataset_Treino_F1_2025_FLAGS.csv'


def add_flags(input_path=INPUT, output_path=OUTPUT):
    df = pd.read_csv(input_path)

    # first lap
    df['is_first_lap'] = df['LapNumber'] == 1

    # first lap of stint (TyreLife == 1)
    df['is_first_stint_lap'] = df['TyreLife'] == 1

    # progress flags de corrida e stint
    total_laps = df.groupby(['Round', 'GP'])['LapNumber'].transform('max').replace(0, np.nan)
    df['race_lap_progress'] = (df['LapNumber'] / total_laps).fillna(0.0).clip(0.0, 1.0)
    df['is_final_quarter'] = df['race_lap_progress'] >= 0.75
    max_stint_life = df.groupby(['Round', 'GP', 'Driver', 'Stint'])['TyreLife'].transform('max').replace(0, np.nan)
    df['stint_progress'] = (df['TyreLife'] / max_stint_life).fillna(0.0).clip(0.0, 1.0)

    # previous was pit (group by GP+Driver sorted by LapNumber)
    df = df.sort_values(['Round', 'GP', 'Driver', 'LapNumber']).reset_index(drop=True)
    df['prev_was_pit'] = False
    grp = df.groupby(['Round','GP','Driver'])
    for name, g in grp:
        idx = g.index
        prev_pit = g['IsPitStop'].shift(fill_value=False)
        df.loc[idx, 'prev_was_pit'] = prev_pit.fillna(False).values

    # is_outlier per driver using robust z-score (median + MAD)
    df['is_outlier'] = False
    for driver, g in df.groupby('Driver'):
        times = g['LapTimeSeconds']
        med = times.median()
        mad = (abs(times - med)).median()
        if mad == 0 or pd.isna(mad):
            z = (times - med) * 0.0
        else:
            z = 0.6745 * (times - med) / mad
        mask = (abs(z) > 3)
        df.loc[g.index, 'is_outlier'] = mask

    # safety car lap heuristic: detecta desaceleracao coletiva por volta
    med_by_gp_lap = df.groupby(['Round','GP','LapNumber'])['LapTimeSeconds'].median().reset_index()
    med_by_gp_lap = med_by_gp_lap.sort_values(['Round', 'GP', 'LapNumber']).reset_index(drop=True)
    med_by_gp_lap['rolling_base'] = (
        med_by_gp_lap
        .groupby(['Round', 'GP'])['LapTimeSeconds']
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    )
    med_by_gp_lap['median_ratio'] = med_by_gp_lap['LapTimeSeconds'] / (med_by_gp_lap['rolling_base'] + 1e-9)
    med_by_gp_lap['median_delta'] = med_by_gp_lap['LapTimeSeconds'] - med_by_gp_lap['rolling_base']

    def mark_sc_candidates(g):
        g = g.copy()
        positive_delta = g.loc[g['median_delta'] > 0, 'median_delta']
        if positive_delta.empty:
            g['sc_flag'] = False
            return g

        dyn_threshold = max(2.5, positive_delta.quantile(0.90))
        g['sc_flag'] = (g['median_ratio'] > 1.06) & (g['median_delta'] >= dyn_threshold)

        # Fallback: se nenhum candidato for encontrado, marca o maior pico real da corrida
        if not g['sc_flag'].any():
            peak_idx = g['median_delta'].idxmax()
            if g.loc[peak_idx, 'median_delta'] > 2.5:
                g.loc[peak_idx, 'sc_flag'] = True
        return g

    med_by_gp_lap = (
        med_by_gp_lap
        .groupby(['Round', 'GP'], group_keys=False)
        .apply(mark_sc_candidates)
        .reset_index(drop=True)
    )

    sc_laps = med_by_gp_lap[med_by_gp_lap['sc_flag']][['Round','GP','LapNumber']]
    sc_laps['sc_flag'] = True

    df = df.merge(sc_laps, on=['Round','GP','LapNumber'], how='left')
    df['safety_car'] = df['sc_flag'].fillna(False)
    df.drop(columns=['sc_flag'], inplace=True)

    # contexto de pista com base no clima agregado
    if 'is_wet_session' in df.columns:
        df['track_context'] = np.where(df['is_wet_session'], 'WET', 'DRY')
    else:
        df['track_context'] = 'UNKNOWN'

    df.to_csv(output_path, index=False)
    print(f'Flags adicionadas e arquivo salvo: {output_path} ({len(df)} linhas)')
    return df

if __name__ == '__main__':
    add_flags()
