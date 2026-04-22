"""
Prepara datasets train/val/test para treino de ML.

Entrada padrao:
    Dataset_Treino_F1_2025_DENOISED.csv

Saidas:
    ml_ready/train.csv
    ml_ready/val.csv
    ml_ready/test.csv
    ml_ready/metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

INPUT_PATH = "Dataset_Treino_F1_2025_DENOISED.csv"
OUTPUT_DIR = "ml_ready"
TARGET_COL = "LapTimeSeconds"


# Features com potencial de leakage direto/indireto para regressao de tempo de volta
LEAKAGE_COLUMNS = {
    "LapTimeSeconds",
    "Sector1TimeSeconds",
    "Sector2TimeSeconds",
    "Sector3TimeSeconds",
}

# IDs / colunas pouco uteis para modelagem direta
ID_COLUMNS = {
    "FullName",
    "Abbreviation",
    "TeamColor",
}


def build_feature_list(df: pd.DataFrame) -> list[str]:
    excluded = set(LEAKAGE_COLUMNS) | set(ID_COLUMNS)
    return [c for c in df.columns if c not in excluded]


def run_precheck(df: pd.DataFrame, features: list[str]) -> dict:
    errors = []
    warnings = []

    if TARGET_COL not in df.columns:
        errors.append(f"Target ausente: {TARGET_COL}")

    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        errors.append(f"Features ausentes: {missing_features}")

    target_nulls = int(df[TARGET_COL].isna().sum()) if TARGET_COL in df.columns else -1
    if target_nulls > 0:
        errors.append(f"Target com nulos: {target_nulls}")

    non_positive_target = int((df[TARGET_COL] <= 0).sum()) if TARGET_COL in df.columns else -1
    if non_positive_target > 0:
        errors.append(f"Target <= 0 encontrado: {non_positive_target}")

    # Checa nulos em features
    feature_nulls = {c: int(df[c].isna().sum()) for c in features}
    high_nulls = {k: v for k, v in feature_nulls.items() if v > 0}
    if high_nulls:
        warnings.append(
            f"Features com nulos (sera necessario imputar no treino): {high_nulls}"
        )

    # integridade de chave
    key_cols = ["Round", "GP", "Driver", "LapNumber"]
    if all(col in df.columns for col in key_cols):
        dup = int(df.duplicated(key_cols).sum())
        if dup > 0:
            warnings.append(f"Duplicatas de chave {key_cols}: {dup}")

    return {
        "errors": errors,
        "warnings": warnings,
        "target_nulls": target_nulls,
        "non_positive_target": non_positive_target,
    }


def split_by_round(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal por Round:
      - train: 1-12
      - val: 13-14
      - test: 15-16
    """
    if "Round" not in df.columns:
        raise ValueError("Coluna 'Round' ausente para split temporal.")

    train = df[df["Round"] <= 12].copy()
    val = df[(df["Round"] >= 13) & (df["Round"] <= 14)].copy()
    test = df[df["Round"] >= 15].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("Split resultou em conjunto vazio (train/val/test).")

    return train, val, test


def main(input_path: str = INPUT_PATH, output_dir: str = OUTPUT_DIR) -> None:
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")

    df = pd.read_csv(in_path)
    features = build_feature_list(df)
    precheck = run_precheck(df, features)

    if precheck["errors"]:
        raise RuntimeError(f"Precheck falhou: {precheck['errors']}")

    train, val, test = split_by_round(df)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    metadata = {
        "input_path": str(in_path),
        "output_dir": str(out_dir),
        "target": TARGET_COL,
        "feature_count": len(features),
        "features": features,
        "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
        "id_columns_excluded": sorted(ID_COLUMNS),
        "rows": {
            "full": int(len(df)),
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
        "rounds": {
            "train": sorted(train["Round"].unique().tolist()),
            "val": sorted(val["Round"].unique().tolist()),
            "test": sorted(test["Round"].unique().tolist()),
        },
        "precheck": precheck,
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=True, indent=2)

    print("ML dataset preparado com sucesso.")
    print(f" - train: {len(train)} linhas")
    print(f" - val:   {len(val)} linhas")
    print(f" - test:  {len(test)} linhas")
    if precheck["warnings"]:
        print(" - warnings:")
        for w in precheck["warnings"]:
            print(f"   * {w}")


if __name__ == "__main__":
    main()
