"""
Valida o dataset agregando tempos por (GP, Driver) e comparando com arquivos
em Track_Results/*_results.csv que possuem FullName e ClassifiedPosition.
Gera um pequeno relatório de concordância.
"""

import pandas as pd
import glob
import os

INPUT = 'Dataset_Treino_F1_2025_DENOISED.csv'
TRACK_RESULTS_DIR = 'Track_Results'
REPORT_PATH = 'official_validation_report.txt'


def validate(input_path=INPUT, results_dir=TRACK_RESULTS_DIR, report_path=REPORT_PATH):
    df = pd.read_csv(input_path)

    agg = df.groupby(['Round', 'GP', 'Driver', 'FullName'], as_index=False)['LapTimeSeconds'].sum()

    total_exact = 0
    total_top3 = 0
    total_abs_error = 0
    total_compared = 0
    rounds_done = 0

    for f in glob.glob(os.path.join(results_dir, '*_results.csv')):
        base = os.path.basename(f)
        if not base.startswith('Round_'):
            continue
        try:
            rnd = int(base.split('_')[1])
        except Exception:
            continue

        resdf = pd.read_csv(f)
        if 'FullName' not in resdf.columns or 'ClassifiedPosition' not in resdf.columns:
            continue

        sub = agg[agg['Round'] == rnd].copy()
        if sub.empty:
            continue

        sub = sub.sort_values('LapTimeSeconds').reset_index(drop=True)
        sub['PredPos'] = range(1, len(sub) + 1)
        pred_map = dict(zip(sub['FullName'], sub['PredPos']))

        compared_round = 0
        for _, row in resdf.iterrows():
            fullname = row['FullName']
            try:
                true_pos = int(row['ClassifiedPosition'])
            except Exception:
                continue
            if fullname not in pred_map:
                continue

            pred_pos = int(pred_map[fullname])
            compared_round += 1
            total_compared += 1
            total_abs_error += abs(true_pos - pred_pos)
            if true_pos == pred_pos:
                total_exact += 1
            if true_pos <= 3 and pred_pos <= 3:
                total_top3 += 1

        if compared_round > 0:
            rounds_done += 1

    exact_acc = (total_exact / total_compared * 100) if total_compared else 0.0
    top3_acc = (total_top3 / total_compared * 100) if total_compared else 0.0
    mae_pos = (total_abs_error / total_compared) if total_compared else 0.0

    lines = [
        'VALIDACAO OFICIAL',
        '-----------------',
        f'Rounds avaliados: {rounds_done}',
        f'Pilotos comparados: {total_compared}',
        f'Exact position match: {exact_acc:.2f}% ({total_exact}/{total_compared})',
        f'Top-3 concordancia: {top3_acc:.2f}% ({total_top3}/{total_compared})',
        f'MAE de posicao: {mae_pos:.3f}',
    ]

    for line in lines:
        print(line)

    with open(report_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines) + '\n')
    print(f'Report salvo em: {report_path}')
    return {
        'exact_acc': exact_acc,
        'top3_acc': top3_acc,
        'mae_pos': mae_pos,
        'total_compared': total_compared,
        'rounds_done': rounds_done,
    }

if __name__ == '__main__':
    validate()
