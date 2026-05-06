"""
Edge case validation report
Checks for:
- missing FullName mappings
- inconsistent TyreLife resets
- laps with LapNumber duplicated per driver
- extreme TyreLife values
Generates a short text report.
"""

import pandas as pd

INPUT = 'Dataset_Treino_F1_2025_DENOISED.csv'
REPORT_PATH = 'edge_case_report.txt'


def run_checks(input_path=INPUT, report_path=REPORT_PATH):
    df = pd.read_csv(input_path)
    report = []

    # Missing FullName
    missing_fullname = df['FullName'].isna().sum()
    report.append(f'Missing FullName: {missing_fullname} rows')

    # Duplicate lap numbers per driver + GP
    dup = df.duplicated(subset=['Round','GP','Driver','LapNumber']).sum()
    report.append(f'Duplicate (Round,GP,Driver,LapNumber): {dup}')

    # TyreLife anomalies (negative or very large)
    neg_tl = (df['TyreLife'] < 0).sum()
    large_tl = (df['TyreLife'] > 100).sum()
    report.append(f'Negative TyreLife: {neg_tl}, TyreLife>100: {large_tl}')
    report.append(f'Missing Compound: {df["Compound"].isna().sum()}')
    report.append(f'Non-positive LapTimeSeconds: {(df["LapTimeSeconds"] <= 0).sum()}')

    # Check TyreLife reset pattern per stint
    anomalies = 0
    tyre_regression = 0
    for _, g in df.groupby(['Round','GP','Driver']):
        g = g.sort_values('LapNumber')
        if g.empty: continue
        if (g['TyreLife'] == 1).sum() == 0:
            anomalies += 1
        if (g['TyreLife'].diff().dropna() < -1).any():
            tyre_regression += 1
    report.append(f'Driver-race with no TyreLife==1 observed: {anomalies}')
    report.append(f'Driver-race with TyreLife regression <-1: {tyre_regression}')

    # Print report
    print('\nEDGE CASES REPORT')
    print('-----------------')
    for r in report:
        print(' -', r)

    with open(report_path, 'w', encoding='utf-8') as fp:
        fp.write('EDGE CASES REPORT\n')
        fp.write('-----------------\n')
        for r in report:
            fp.write(f'- {r}\n')
    print(f'\nReport salvo em: {report_path}')
    return report

if __name__ == '__main__':
    run_checks()
