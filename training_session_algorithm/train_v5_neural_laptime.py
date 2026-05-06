#!/usr/bin/env python3
"""
Train V5 Neural LapTime — Motor de corrida neural treinado com dados reais
==========================================================================

- Treina um modelo supervisionado (LapTimePredictor) com Dataset_Treino_F1_2025_BALANCED.csv
- O modelo neural substitui as fórmulas paramétricas de tempo de volta
- O ambiente RL usa esse modelo como "motor físico" da corrida
- Observation space: 14 features (igual ao V4)
- O agente RL aprende estratégia de pit stop contra um ambiente que reflete
  exatamente os dados reais de 2025

Pipeline:
  1. Descoberta de todos os dados do projeto
  2. Treino supervisionado do LapTimePredictor (sklearn GradientBoosting)
  3. Treino RL (PPO) usando o predictor como motor do ambiente

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training_v5_neural.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================

COMPOUND_MAP   = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
COMPOUND_NAMES = {v: k for k, v in COMPOUND_MAP.items()}
COMPOUND_MAIN  = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}   # só compostos secos principais
MANDATORY_COMPOUNDS = 2
PIT_STOP_TIME_LOSS  = 23.5
GRID_GAP_SECONDS    = 1.8
SC_PIT_DISCOUNT     = 0.35

TRACK_PASSING_FACTOR = {
    "Monaco Grand Prix": 1.45, "Hungarian Grand Prix": 1.20,
    "Singapore Grand Prix": 1.18, "São Paulo Grand Prix": 1.00,
    "Austrian Grand Prix": 0.95, "Bahrain Grand Prix": 0.92,
    "Belgian Grand Prix": 0.88, "Italian Grand Prix": 0.85,
    "Saudi Arabian Grand Prix": 0.90, "Australian Grand Prix": 0.93,
    "Japanese Grand Prix": 0.96, "Chinese Grand Prix": 0.94,
    "Miami Grand Prix": 0.91, "Emilia Romagna Grand Prix": 0.97,
    "Spanish Grand Prix": 0.98, "Canadian Grand Prix": 0.89,
    "British Grand Prix": 0.87, "Dutch Grand Prix": 0.99,
    "Azerbaijan Grand Prix": 1.10, "United States Grand Prix": 0.86,
    "Mexico City Grand Prix": 0.92, "Las Vegas Grand Prix": 0.84,
    "Qatar Grand Prix": 0.88, "Abu Dhabi Grand Prix": 0.90,
}

TEAM_NAME_MAP = {
    "Red Bull": "Red Bull Racing", "Racing Bulls": "RB",
    "Sauber": "Kick Sauber", "Haas": "Haas F1 Team",
    "McLaren": "McLaren", "Mercedes": "Mercedes",
    "Ferrari": "Ferrari", "Aston Martin": "Aston Martin",
    "Alpine": "Alpine", "Williams": "Williams",
}

DRIVER_CODES = {
    "Max Verstappen": "VER", "Lando Norris": "NOR", "Lewis Hamilton": "HAM",
    "Charles Leclerc": "LEC", "George Russell": "RUS", "Oscar Piastri": "PIA",
    "Kimi Antonelli": "ANT", "Fernando Alonso": "ALO", "Carlos Sainz": "SAI",
    "Isack Hadjar": "HAD", "Gabriel Bortoleto": "BOR", "Oliver Bearman": "BEA",
    "Nico Hulkenberg": "HUL", "Alexander Albon": "ALB", "Pierre Gasly": "GAS",
    "Yuki Tsunoda": "TSU", "Esteban Ocon": "OCO", "Franco Colapinto": "COL",
    "Liam Lawson": "LAW", "Lance Stroll": "STR",
}

# ============================================================================
# DESCOBERTA DE DADOS
# ============================================================================

def find_project_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "f1_data").exists() or (candidate / "ml_ready_data").exists():
            return candidate
    return start


def discover_data_paths(data_root: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    script_dir = Path(__file__).resolve().parent
    root = Path(data_root) if data_root else find_project_root(script_dir)

    def find(candidates: List[str]) -> Optional[Path]:
        for rel in candidates:
            p = root / rel
            if p.exists():
                return p
        return None

    _DS_DIR = "ml_prepare_data/ml_thinking/python_scripts_algorotimo_prep,trei,val"

    paths = {
        "f1_data":           find(["f1_data", "RL_F1_for_performance/f1_data"]),
        "lap_data":          find(["f1_data/lap_data.csv", "RL_F1_for_performance/f1_data/lap_data.csv"]),
        "pit_stops":         find(["f1_data/pit_stops.csv", "RL_F1_for_performance/f1_data/pit_stops.csv"]),
        "race_events":       find(["f1_data/race_events.csv", "RL_F1_for_performance/f1_data/race_events.csv"]),
        "team_performances": find(["f1_data/team_performances.csv", "RL_F1_for_performance/f1_data/team_performances.csv"]),
        "drivers":           find(["f1_data/drivers.csv", "RL_F1_for_performance/f1_data/drivers.csv"]),
        "track_results_dir": find([
            "ml_prepare_data/ml_thinking/Track_Results",
            "ml_thinking/Track_Results",
            "Track_Results",
        ]),
        "dataset_balanced":  find([
            f"{_DS_DIR}/Dataset_Treino_F1_2025_BALANCED.csv",
            "ml_thinking/python_scripts_algorotimo_prep,trei,val/Dataset_Treino_F1_2025_BALANCED.csv",
        ]),
        "dataset_cleaned":   find([
            f"{_DS_DIR}/Dataset_Treino_F1_2025_CLEANED.csv",
            "ml_thinking/python_scripts_algorotimo_prep,trei,val/Dataset_Treino_F1_2025_CLEANED.csv",
        ]),
        "dataset_denoised":  find([
            f"{_DS_DIR}/Dataset_Treino_F1_2025_DENOISED.csv",
            "ml_thinking/python_scripts_algorotimo_prep,trei,val/Dataset_Treino_F1_2025_DENOISED.csv",
        ]),
        "dataset_flags":     find([
            f"{_DS_DIR}/Dataset_Treino_F1_2025_FLAGS.csv",
            "ml_thinking/python_scripts_algorotimo_prep,trei,val/Dataset_Treino_F1_2025_FLAGS.csv",
        ]),
        "dataset_base":      find([
            f"{_DS_DIR}/Dataset_Treino_F1_2025.csv",
            "ml_thinking/python_scripts_algorotimo_prep,trei,val/Dataset_Treino_F1_2025.csv",
        ]),
        "train_split":       find([
            "ml_prepare_data/ml_ready_data/ml_ready/train.csv",
            "ml_prepare_data/ml_ready_data/train.csv",
            "ml_ready_data/Circuit_List/train.csv",
        ]),
        "val_split":         find([
            "ml_prepare_data/ml_ready_data/ml_ready/val.csv",
            "ml_prepare_data/ml_ready_data/val.csv",
            "ml_ready_data/Circuit_List/val.csv",
        ]),
        "test_split":        find([
            "ml_prepare_data/ml_ready_data/ml_ready/test.csv",
            "ml_prepare_data/ml_ready_data/test.csv",
            "ml_ready_data/Circuit_List/test.csv",
        ]),
        "telemetry_master":  find([
            "ml_prepare_data/ml_ready_data/Circuit_List/F1_2025_Telemetry_Master.csv",
            "ml_ready_data/Circuit_List/F1_2025_Telemetry_Master.csv",
        ]),
    }

    logger.info("=== Fontes de dados ===")
    for k, v in paths.items():
        logger.info(f"  {'✅' if v else '❌'} {k}: {v}")
    logger.info("=======================")
    return paths


# ============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ML
# ============================================================================

# Mapeamentos de colunas possíveis nos datasets
_COL_ALIASES = {
    "laptime":    ["LapTime_s", "LapTime", "laptime", "lap_time", "Time", "time_s", "LapTimeSeconds"],
    "lapnumber":  ["LapNumber", "lap_number", "Lap", "lap", "LapNum"],
    "compound":   ["Compound", "compound", "TyreCompound", "tyre_compound"],
    "tyrelife":   ["TyreLife", "tyre_life", "TireLife", "TyreAge", "laps_on_tyre"],
    "driver":     ["Driver", "driver", "DriverName", "driver_name", "Abbreviation"],
    "team":       ["Team", "team", "Constructor", "TeamName", "constructor"],
    "racename":   ["RaceName", "race_name", "Race", "GrandPrix", "grand_prix", "EventName"],
    "position":   ["Position", "position", "ClassifiedPosition", "pos", "Pos"],
    "grid":       ["Grid", "grid", "GridPosition", "StartingGrid"],
    "safety_car": ["SafetyCar", "safety_car", "SC", "TrackStatus"],
    "track_temp": ["TrackTemp", "track_temp", "AirTemp", "air_temp", "Temperature"],
    "speed":      ["SpeedST", "SpeedFL", "speed", "Speed", "TopSpeed"],
}


def _find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    """Retorna o nome real da coluna no DataFrame para um alias lógico."""
    for alias in _COL_ALIASES.get(key, []):
        if alias in df.columns:
            return alias
    return None


def _canonical_compound(value: str) -> int:
    """Mapeia string de composto para inteiro."""
    v = str(value).upper().strip()
    for k, idx in COMPOUND_MAP.items():
        if k in v:
            return idx
    return COMPOUND_MAP["MEDIUM"]


def load_and_merge_all_datasets(paths: Dict[str, Optional[Path]]) -> pd.DataFrame:
    """
    Carrega e une todos os datasets disponíveis em um único DataFrame.
    Estratégia: usa BALANCED como base, enriquece com FLAGS e DENOISED,
    adiciona train/val/test splits, e por último telemetria.
    """
    frames = []

    priority_order = [
        "dataset_balanced",
        "dataset_denoised",
        "dataset_cleaned",
        "dataset_base",
        "train_split",
        "val_split",
        "test_split",
    ]

    seen_files = set()
    for key in priority_order:
        p = paths.get(key)
        if p and p.exists() and str(p) not in seen_files:
            try:
                df = pd.read_csv(p, low_memory=False)
                df["_source"] = key
                frames.append(df)
                seen_files.add(str(p))
                logger.info(f"  📄 {p.name}: {len(df):,} linhas × {len(df.columns)} colunas")
            except Exception as e:
                logger.warning(f"  ⚠️  Erro {p.name}: {e}")

    # Track results individuais
    track_dir = paths.get("track_results_dir")
    if track_dir and track_dir.exists():
        for f in sorted(track_dir.glob("*.csv")):
            try:
                df = pd.read_csv(f, low_memory=False)
                df["_source"] = "track_result"
                frames.append(df)
                seen_files.add(str(f))
            except Exception:
                pass
        logger.info(f"  📄 Track_Results: {len(list(track_dir.glob('*.csv')))} arquivos")

    # Telemetria (pode ser grande — carrega seletivamente)
    tel_path = paths.get("telemetry_master")
    if tel_path and tel_path.exists():
        try:
            # Lê só as primeiras 200k linhas pra não explodir memória
            df = pd.read_csv(tel_path, low_memory=False, nrows=200_000)
            df["_source"] = "telemetry"
            frames.append(df)
            logger.info(f"  📄 Telemetry Master: {len(df):,} linhas (amostra)")
        except Exception as e:
            logger.warning(f"  ⚠️  Telemetria: {e}")

    if not frames:
        raise RuntimeError("Nenhum dataset ML encontrado. Verifique --data-root.")

    # Concatena com colunas em comum
    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info(f"✅ Dataset combinado: {len(combined):,} linhas × {len(combined.columns)} colunas")
    return combined


# ============================================================================
# MODELO SUPERVISIONADO DE LAP TIME
# ============================================================================

class LapTimePredictor:
    """
    Modelo supervisionado (GradientBoosting) que prediz tempos de volta
    a partir de features reais do dataset 2025.

    Features de entrada:
        - compound_id      : int 0-4
        - tyre_life        : int (voltas no pneu)
        - lap_number       : int
        - lap_progress     : float (lap / total_laps)
        - position         : int
        - safety_car       : int 0/1
        - team_gap         : float (segundos vs melhor equipe)
        - driver_racepace  : float (score 0-100)
        - driver_tyremgmt  : float (score 0-100)
        - track_factor     : float (dificuldade ultrapassagem)

    Saída: LapTime_s (float, segundos)
    """

    FEATURES = [
        "compound_id", "tyre_life", "lap_number", "lap_progress",
        "position", "safety_car", "team_gap",
        "driver_racepace", "driver_tyremgmt", "track_factor",
    ]

    def __init__(self):
        self.model   = None
        self.scaler  = None
        self.is_trained = False
        self.fallback_params: Dict = {}   # parâmetros estatísticos de fallback

    # ------------------------------------------------------------------
    # Treino supervisionado
    # ------------------------------------------------------------------

    def train(
        self,
        combined_df: pd.DataFrame,
        lap_df: pd.DataFrame,
        team_perf_df: pd.DataFrame,
        drivers_df: pd.DataFrame,
    ) -> Dict:
        """
        Prepara features a partir dos dados reais e treina o modelo.
        Retorna métricas de avaliação.
        """
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        logger.info("🔧 Preparando features para LapTimePredictor...")

        # --- Determina colunas ---
        laptime_col  = _find_col(combined_df, "laptime")
        lapnum_col   = _find_col(combined_df, "lapnumber")
        compound_col = _find_col(combined_df, "compound")
        tyrelife_col = _find_col(combined_df, "tyrelife")
        driver_col   = _find_col(combined_df, "driver")
        team_col     = _find_col(combined_df, "team")
        racename_col = _find_col(combined_df, "racename")
        pos_col      = _find_col(combined_df, "position")
        sc_col       = _find_col(combined_df, "safety_car")

        # Também tenta pegar do lap_df base (mais confiável)
        lap_laptime  = _find_col(lap_df, "laptime")
        lap_lapnum   = _find_col(lap_df, "lapnumber")
        lap_compound = _find_col(lap_df, "compound")
        lap_tyrelife = _find_col(lap_df, "tyrelife")
        lap_driver   = _find_col(lap_df, "driver")
        lap_race     = _find_col(lap_df, "racename")

        # --- Monta DataFrame de treino ---
        rows = []

        # Constrói lookups de equipe e piloto
        team_gap_map = self._build_team_gap_map(team_perf_df)
        driver_map   = self._build_driver_map(drivers_df)

        # Calcula total_laps por corrida do lap_df
        total_laps_map: Dict[str, int] = {}
        if lap_lapnum and lap_race:
            for race, grp in lap_df.groupby(lap_race):
                total_laps_map[str(race)] = int(grp[lap_lapnum].max())

        # Processa lap_df (fonte primária de lap times)
        if all([lap_laptime, lap_lapnum, lap_compound, lap_tyrelife, lap_driver, lap_race]):
            logger.info(f"  → Processando lap_df ({len(lap_df):,} linhas)...")
            for _, row in lap_df.iterrows():
                lt = self._safe_float(row.get(lap_laptime))
                if lt is None or lt < 60 or lt > 300:   # filtra outliers
                    continue
                race     = str(row.get(lap_race, ""))
                driver   = str(row.get(lap_driver, ""))
                cmp_str  = str(row.get(lap_compound, "MEDIUM"))
                lap_num  = self._safe_int(row.get(lap_lapnum), 1)
                tyre_age = self._safe_int(row.get(lap_tyrelife), 1)
                total_laps = total_laps_map.get(race, 60)
                dp = driver_map.get(driver, {})
                rows.append({
                    "compound_id":     _canonical_compound(cmp_str),
                    "tyre_life":       min(tyre_age, 60),
                    "lap_number":      lap_num,
                    "lap_progress":    lap_num / max(total_laps, 1),
                    "position":        10,   # desconhecido nessa fonte
                    "safety_car":      0,
                    "team_gap":        team_gap_map.get(dp.get("team", ""), 1.0),
                    "driver_racepace": dp.get("racepace", 85.0),
                    "driver_tyremgmt": dp.get("tyremgmt", 85.0),
                    "track_factor":    TRACK_PASSING_FACTOR.get(race, 1.0),
                    "laptime_s":       lt,
                })

        # Processa combined_df (datasets ML processados)
        if laptime_col and lapnum_col:
            logger.info(f"  → Processando combined_df ({len(combined_df):,} linhas)...")
            for _, row in combined_df.iterrows():
                lt = self._safe_float(row.get(laptime_col) if laptime_col else None)
                if lt is None or lt < 60 or lt > 300:
                    continue
                race     = str(row.get(racename_col, "")) if racename_col else ""
                driver   = str(row.get(driver_col, "")) if driver_col else ""
                cmp_str  = str(row.get(compound_col, "MEDIUM")) if compound_col else "MEDIUM"
                lap_num  = self._safe_int(row.get(lapnum_col), 1)
                tyre_age = self._safe_int(row.get(tyrelife_col), 1) if tyrelife_col else 1
                total_laps = total_laps_map.get(race, 60)
                dp = driver_map.get(driver, {})
                # Safety car
                sc = 0
                if sc_col:
                    sc_val = str(row.get(sc_col, "")).upper()
                    sc = 1 if any(x in sc_val for x in ["SC", "SAFETY", "1", "TRUE", "VSC"]) else 0
                # Posição
                pos = self._safe_int(row.get(pos_col), 10) if pos_col else 10
                rows.append({
                    "compound_id":     _canonical_compound(cmp_str),
                    "tyre_life":       min(tyre_age, 60),
                    "lap_number":      lap_num,
                    "lap_progress":    lap_num / max(total_laps, 1),
                    "position":        min(pos, 20),
                    "safety_car":      sc,
                    "team_gap":        team_gap_map.get(dp.get("team", ""), 1.0),
                    "driver_racepace": dp.get("racepace", 85.0),
                    "driver_tyremgmt": dp.get("tyremgmt", 85.0),
                    "track_factor":    TRACK_PASSING_FACTOR.get(race, 1.0),
                    "laptime_s":       lt,
                })

        if len(rows) < 100:
            logger.warning(f"⚠️  Apenas {len(rows)} amostras válidas. Usando modelo de fallback.")
            self._build_fallback(lap_df)
            return {"status": "fallback", "samples": len(rows)}

        train_data = pd.DataFrame(rows)
        logger.info(f"✅ Dataset de treino supervisionado: {len(train_data):,} amostras")

        X = train_data[self.FEATURES].values
        y = train_data["laptime_s"].values

        # Salva parâmetros de fallback
        self.fallback_params = {
            "mean": float(np.mean(y)),
            "std":  float(np.std(y)),
            "compound_deltas": {
                c: float(train_data[train_data["compound_id"] == cid]["laptime_s"].mean() - np.mean(y))
                for c, cid in COMPOUND_MAIN.items()
            },
        }

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

        # Scaler
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_s   = self.scaler.fit_transform(X_train)
        X_val_s     = self.scaler.transform(X_val)

        logger.info("🤖 Treinando GradientBoostingRegressor...")
        self.model = GradientBoostingRegressor(
            n_estimators     = 300,
            learning_rate    = 0.08,
            max_depth        = 5,
            min_samples_leaf = 20,
            subsample        = 0.8,
            random_state     = 42,
            verbose          = 0,
        )
        self.model.fit(X_train_s, y_train)

        # Avaliação
        y_pred_val = self.model.predict(X_val_s)
        mae        = mean_absolute_error(y_val, y_pred_val)
        r2         = r2_score(y_val, y_pred_val)
        self.is_trained = True

        metrics = {
            "status":        "trained",
            "samples":       len(train_data),
            "train_samples": len(X_train),
            "val_samples":   len(X_val),
            "mae_seconds":   round(mae, 4),
            "r2_score":      round(r2, 4),
        }
        logger.info(f"✅ LapTimePredictor treinado! MAE={mae:.3f}s | R²={r2:.4f}")

        # Feature importance
        fi = pd.Series(self.model.feature_importances_, index=self.FEATURES).sort_values(ascending=False)
        logger.info("📊 Feature importance (top 5):")
        for feat, imp in fi.head(5).items():
            logger.info(f"    {feat}: {imp:.4f}")

        return metrics

    def predict(
        self,
        compound_id: int,
        tyre_life: int,
        lap_number: int,
        lap_progress: float,
        position: int,
        safety_car: int,
        team_gap: float,
        driver_racepace: float,
        driver_tyremgmt: float,
        track_factor: float,
        noise_sigma: float = 0.04,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Prediz tempo de volta. Adiciona ruído gaussiano para variabilidade."""
        if rng is None:
            rng = np.random.default_rng()

        # Safety car: adiciona ~32s artificialmente
        sc_penalty = 32.0 if safety_car else 0.0

        if self.is_trained and self.model is not None and self.scaler is not None:
            X = np.array([[
                compound_id, tyre_life, lap_number, lap_progress,
                position, safety_car, team_gap,
                driver_racepace, driver_tyremgmt, track_factor,
            ]], dtype=np.float32)
            X_s     = self.scaler.transform(X)
            base_lt = float(self.model.predict(X_s)[0])
        else:
            # Fallback paramétrico
            base_lt = self.fallback_params.get("mean", 95.0)
            compound_delta = {0: 0.0, 1: 0.35, 2: 0.75}.get(compound_id, 0.35)
            base_lt += compound_delta
            base_lt += tyre_life * 0.08   # degradação simples
            base_lt += (position - 1) * 0.04 * track_factor
            base_lt += team_gap
            base_lt += (99.0 - driver_racepace) * 0.04

        noise = float(rng.normal(0.0, noise_sigma))
        return float(base_lt + sc_penalty + noise)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":           self.model,
                "scaler":          self.scaler,
                "is_trained":      self.is_trained,
                "fallback_params": self.fallback_params,
            }, f)
        logger.info(f"💾 LapTimePredictor salvo em: {path}")

    @classmethod
    def load(cls, path: Path) -> "LapTimePredictor":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model           = data["model"]
        obj.scaler          = data["scaler"]
        obj.is_trained      = data["is_trained"]
        obj.fallback_params = data["fallback_params"]
        logger.info(f"📥 LapTimePredictor carregado de: {path}")
        return obj

    # ------------------------------------------------------------------
    # Utilitários internos
    # ------------------------------------------------------------------

    def _build_team_gap_map(self, team_perf_df: pd.DataFrame) -> Dict[str, float]:
        team_perf_df = team_perf_df.copy()
        team_perf_df["TeamKey"] = team_perf_df["Team"].astype(str).str.strip().map(
            lambda x: TEAM_NAME_MAP.get(x, x))
        ref = float(team_perf_df["AvgLapTime_s"].min())
        return {row["TeamKey"]: float(row["AvgLapTime_s"] - ref)
                for _, row in team_perf_df.iterrows()}

    def _build_driver_map(self, drivers_df: pd.DataFrame) -> Dict[str, Dict]:
        result = {}
        for _, row in drivers_df.iterrows():
            driver = str(row["Driver"]).strip()
            team   = TEAM_NAME_MAP.get(str(row["Team"]).strip(), str(row["Team"]).strip())
            result[driver] = {
                "team":      team,
                "racepace":  float(row.get("RacePace", 85.0)),
                "tyremgmt":  float(row.get("TyreManagement", 85.0)),
                "overall":   float(row.get("Overall", 85.0)),
            }
        return result

    def _build_fallback(self, lap_df: pd.DataFrame):
        """Parâmetros estatísticos simples quando não há dados suficientes."""
        lt_col = _find_col(lap_df, "laptime")
        if lt_col:
            valid = pd.to_numeric(lap_df[lt_col], errors="coerce").dropna()
            valid = valid[(valid > 60) & (valid < 300)]
            self.fallback_params["mean"] = float(valid.mean()) if not valid.empty else 95.0
            self.fallback_params["std"]  = float(valid.std())  if not valid.empty else 1.5
        else:
            self.fallback_params["mean"] = 95.0
            self.fallback_params["std"]  = 1.5

    @staticmethod
    def _safe_float(v) -> Optional[float]:
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(v, default: int = 1) -> int:
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


# ============================================================================
# DADOS AUXILIARES (Track Results, calibração)
# ============================================================================

def load_track_results(track_results_dir: Optional[Path]) -> Dict[str, pd.DataFrame]:
    result = {}
    if not (track_results_dir and track_results_dir.exists()):
        logger.warning("⚠️  Track_Results não encontrado.")
        return result
    for f in sorted(track_results_dir.glob("*.csv")):
        try:
            df   = pd.read_csv(f)
            stem = f.stem
            parts = [p for p in stem.split("_")
                     if p not in {"Round", "results", "2025"} and not p.isdigit()]
            race_name = " ".join(parts).strip()
            if race_name:
                result[race_name] = df
        except Exception as e:
            logger.warning(f"  ⚠️  {f.name}: {e}")
    logger.info(f"Track_Results: {len(result)} corridas")
    return result


def build_track_calibration(track_results: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    cal = {}
    for race, df in track_results.items():
        c = {}
        if "Laps" in df.columns:
            c["total_laps"] = int(pd.to_numeric(df["Laps"], errors="coerce").max())
        if "Stops" in df.columns:
            c["avg_pit_stops"] = float(pd.to_numeric(df["Stops"], errors="coerce").mean())
        if "Grid" in df.columns and "Position" in df.columns:
            grid = pd.to_numeric(df["Grid"], errors="coerce")
            pos  = pd.to_numeric(df["Position"], errors="coerce")
            c["avg_position_change"] = float((grid - pos).mean())
        if "Team" in df.columns and "Position" in df.columns:
            w = df[pd.to_numeric(df["Position"], errors="coerce") == 1]
            if not w.empty:
                c["winning_team"] = str(w.iloc[0]["Team"])
        cal[race] = c
    return cal


def extract_driver_stats(combined_df: pd.DataFrame) -> Dict[str, Dict]:
    driver_col = _find_col(combined_df, "driver")
    if not driver_col:
        return {}
    pos_col  = _find_col(combined_df, "position")
    grid_col = _find_col(combined_df, "grid")
    stats = {}
    for driver, grp in combined_df.groupby(driver_col):
        s = {"race_count": len(grp)}
        if pos_col:
            nums = pd.to_numeric(grp[pos_col], errors="coerce").dropna()
            if not nums.empty:
                s["avg_finish_position"] = float(nums.mean())
                s["best_finish"]         = float(nums.min())
        if grid_col:
            nums = pd.to_numeric(grp[grid_col], errors="coerce").dropna()
            if not nums.empty:
                s["avg_grid"] = float(nums.mean())
        stats[str(driver)] = s
    logger.info(f"Estatísticas ML: {len(stats)} pilotos")
    return stats


# ============================================================================
# AMBIENTE RL V5 — usa LapTimePredictor como motor
# ============================================================================

import gymnasium as gym
from gymnasium import spaces


class F1RaceEnvV5(gym.Env):
    """
    Ambiente F1 V5 com motor de corrida neural.

    O LapTimePredictor (GradientBoosting treinado nos dados reais 2025)
    substitui as fórmulas paramétricas do V3/V4 para calcular tempos de volta.

    Observation Space: Box(14) — igual ao V4
    Action Space:      Discrete(4) — igual ao V4
    """

    metadata = {"render_modes": ["human"]}
    OBS_DIM  = 14

    def __init__(
        self,
        data_paths: Dict[str, Optional[Path]],
        lap_predictor: LapTimePredictor,
        track_calibration: Dict[str, Dict],
        driver_ml_stats: Dict[str, Dict],
        team_gap_map: Dict[str, float],
        drivers_df: pd.DataFrame,
        lap_df: pd.DataFrame,
        pit_df: pd.DataFrame,
        events_df: pd.DataFrame,
        race_name=None,
        agent_team=None,
        agent_driver=None,
        starting_position=None,
        render_mode=None,
    ):
        super().__init__()
        self.lap_predictor     = lap_predictor
        self.track_calibration = track_calibration
        self.driver_ml_stats   = driver_ml_stats
        self.team_gap_map      = team_gap_map
        self.drivers_df        = drivers_df
        self.lap_df            = lap_df
        self.pit_df            = pit_df
        self.events_df         = events_df
        self.race_name         = race_name
        self.agent_team        = agent_team
        self.agent_driver      = agent_driver
        self.starting_position = starting_position
        self.render_mode       = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.zeros(self.OBS_DIM, dtype=np.float32),
            high=np.ones(self.OBS_DIM,  dtype=np.float32),
            dtype=np.float32,
        )

        _race_col_init = _find_col(self.lap_df, "racename") or "RaceName"
        self.available_races = sorted(self.lap_df[_race_col_init].dropna().unique().tolist())
        self._precompute_pit_time()
        self._reset_state()

    def _precompute_pit_time(self):
        pit_col = (
            next((c for c in self.pit_df.columns
                  if any(x in c.lower() for x in ["duration", "pittime", "pit_time", "stop_time"])), None)
            or "PitDuration_s"
        )
        if pit_col in self.pit_df.columns:
            vals = pd.to_numeric(self.pit_df[pit_col], errors="coerce").dropna()
            self.pit_time_loss = max(float(vals.median()) if not vals.empty else PIT_STOP_TIME_LOSS,
                                     PIT_STOP_TIME_LOSS)
        else:
            self.pit_time_loss = PIT_STOP_TIME_LOSS

    def _reset_state(self):
        self.current_lap   = 0
        self.total_laps    = 60
        self.position      = 1
        self.compound      = COMPOUND_MAIN["MEDIUM"]
        self.tyre_life     = 1
        self.safety_car_active = False
        self.compounds_used    = set()
        self.pit_count         = 0
        self.total_time        = 0.0
        self.lap_times         = []
        self._sc_laps          = set()
        self.opponent_times    = {}
        self.num_competitors   = 20
        self.driver_history    = {}
        self.sc_history        = []
        self.grid_order        = []
        self.track_factor      = 1.0
        self._expected_pit_stops = 1.5

    # ------------------------------------------------------------------
    # Perfil de pilotos/equipes
    # ------------------------------------------------------------------

    def _canonical_team(self, t):
        return TEAM_NAME_MAP.get(t, t)

    def _get_driver_profile(self, driver_name=None, team_name=None):
        df = self.drivers_df.copy()
        if driver_name:
            df = df[df["Driver"] == driver_name]
        if team_name:
            df = df[df["TeamKey"] == self._canonical_team(team_name)]
        row = df.iloc[0] if not df.empty else self.drivers_df.iloc[0]
        ml  = self.driver_ml_stats.get(str(row["Driver"]), {})
        return {
            "Driver":          row["Driver"],
            "Team":            row["TeamKey"],
            "RacePace":        float(row.get("RacePace", 85.0)),
            "TyreManagement":  float(row.get("TyreManagement", 85.0)),
            "Overall":         float(row.get("Overall", 85.0)),
            "Code":            DRIVER_CODES.get(row["Driver"], row["Driver"][:3].upper()),
            "AvgFinishPos_ML": float(ml.get("avg_finish_position", 10.0)),
            "TotalPoints_ML":  float(ml.get("total_points", 0.0)),
        }

    def _build_driver_model(self, profile):
        team     = profile["Team"]
        team_gap = self.team_gap_map.get(team, 1.0)
        top_bonus = float(self.np_random.normal(0.0, 0.08)) if team_gap <= 0.15 else 0.0
        noise_sigma = 0.055 if team_gap <= 0.15 else 0.035
        return {
            "team_gap":       team_gap,
            "noise_sigma":    noise_sigma + top_bonus * 0.01,
            "profile":        profile,
        }

    # ------------------------------------------------------------------
    # Safety car
    # ------------------------------------------------------------------

    def _build_safety_car_schedule(self, race_events):
        self._sc_laps = set()
        sc = race_events[race_events["StatusName"].isin(["SafetyCar", "VirtualSafetyCar"])] \
             if "StatusName" in race_events.columns else pd.DataFrame()
        if sc.empty:
            return
        n = len(sc)
        for i in range(n):
            center = int(self.total_laps * (i + 1) / (n + 1))
            for off in range(3):
                lap = center + off
                if 1 <= lap <= self.total_laps:
                    self._sc_laps.add(lap)

    # ------------------------------------------------------------------
    # Estratégia de pneus
    # ------------------------------------------------------------------

    def _choose_start_compound(self, model):
        tg = model["team_gap"]
        if self.total_laps >= 55:
            return COMPOUND_MAIN["MEDIUM"] if tg < 0.30 else COMPOUND_MAIN["HARD"]
        return COMPOUND_MAIN["SOFT"] if tg < 0.15 else COMPOUND_MAIN["MEDIUM"]

    def _stint_target(self, compound, model):
        base = {COMPOUND_MAIN["SOFT"]: 11, COMPOUND_MAIN["MEDIUM"]: 19, COMPOUND_MAIN["HARD"]: 28}[compound]
        return base + int((model["profile"]["TyreManagement"] - 85) * 0.30)

    def _choose_next_compound(self, current, lap, model, used):
        laps_left = self.total_laps - lap
        if laps_left <= 12:
            return COMPOUND_MAIN["SOFT"] if current != COMPOUND_MAIN["SOFT"] else COMPOUND_MAIN["MEDIUM"]
        for c in [COMPOUND_MAIN["HARD"], COMPOUND_MAIN["MEDIUM"], COMPOUND_MAIN["SOFT"]]:
            if c != current and c not in used:
                return c
        return COMPOUND_MAIN["MEDIUM"]

    def _should_pit(self, tyre_life, compound, lap, model, used):
        target    = self._stint_target(compound, model)
        laps_left = self.total_laps - lap + 1
        if tyre_life >= target:
            return True
        if compound == COMPOUND_MAIN["SOFT"] and tyre_life >= max(9, target - 1) and laps_left > 9:
            return True
        if self.safety_car_active and tyre_life >= max(8, int(target * 0.70)) and laps_left > 7:
            return True
        if len(used) < 2 and laps_left <= target:
            return True
        return False

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def _grid_score(self, profile):
        tg = self.team_gap_map.get(profile["Team"], 1.0)
        ml_bonus = (profile.get("AvgFinishPos_ML", 10.0) - 10.0) * 0.015
        return tg + (99.0 - profile["Overall"]) * 0.03 + ml_bonus + float(self.np_random.normal(0, 0.02))

    def _create_grid(self):
        n_drivers = len(self.drivers_df)
        pool = [
            self._get_driver_profile(
                driver_name=self.drivers_df.iloc[
                    int(self.np_random.integers(0, n_drivers))
                ]["Driver"]
            )
            for _ in range(self.num_competitors - 1)
        ]
        for p in pool:
            p["_gs"] = self._grid_score(p)
        pool.sort(key=lambda p: p["_gs"])
        agent_gs = self._grid_score(self.agent_model["profile"])

        if self.starting_position is not None:
            insert = max(1, min(int(self.starting_position), self.num_competitors))
        else:
            insert = 1 + sum(1 for p in pool if p["_gs"] < agent_gs)
            insert = max(1, min(insert, self.num_competitors))

        grid = []
        idx  = 0
        for pos in range(1, self.num_competitors + 1):
            if pos == insert:
                grid.append((pos, self.agent_model["profile"], True))
            elif idx < len(pool):
                grid.append((pos, pool[idx], False))
                idx += 1
        self.grid_order = grid
        self.position   = insert

    def _init_opponents(self):
        self.opponent_times = {}
        self._create_grid()
        for pos, profile, is_agent in self.grid_order:
            if is_agent:
                continue
            m = self._build_driver_model(profile)
            c = self._choose_start_compound(m)
            self.opponent_times[profile["Driver"]] = {
                "total_time":    (pos - 1) * GRID_GAP_SECONDS,
                "model":         m,
                "compound":      c,
                "tyre_life":     1,
                "pit_count":     0,
                "position":      pos,
                "pit_this_lap":  False,
                "compounds_used": {c},
            }
        self.total_time = (self.position - 1) * GRID_GAP_SECONDS

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and isinstance(options, dict):
            self.race_name         = options.get("race_name",         self.race_name)
            self.agent_team        = options.get("agent_team",        self.agent_team)
            self.agent_driver      = options.get("agent_driver",      self.agent_driver)
            self.starting_position = options.get("starting_position", self.starting_position)

        race        = self.race_name if self.race_name else self.np_random.choice(self.available_races)
        _race_col   = _find_col(self.lap_df, "racename") or "RaceName"
        _lap_col    = _find_col(self.lap_df, "lapnumber") or "LapNumber"
        _drv_col    = _find_col(self.lap_df, "driver") or "Driver"
        _ev_race_col = _find_col(self.events_df, "racename") or "RaceName"
        race_laps   = self.lap_df[self.lap_df[_race_col] == race]
        race_events = self.events_df[self.events_df[_ev_race_col] == race]

        self.total_laps      = int(race_laps[_lap_col].max()) if not race_laps.empty else 60
        self.num_competitors = max(int(race_laps[_drv_col].nunique()), 2) if not race_laps.empty else 20
        self.track_factor    = TRACK_PASSING_FACTOR.get(race, 1.0)
        self._build_safety_car_schedule(race_events)

        cal = self.track_calibration.get(race, {})
        self._expected_pit_stops = cal.get("avg_pit_stops", 1.5)
        if cal.get("total_laps", 0) > 0:
            self.total_laps = cal["total_laps"]

        if self.agent_driver:
            prof = self._get_driver_profile(driver_name=self.agent_driver)
        elif self.agent_team:
            prof = self._get_driver_profile(team_name=self.agent_team)
        else:
            prof = self._get_driver_profile()

        self.agent_model    = self._build_driver_model(prof)
        self.current_lap    = 1
        self.compound       = self._choose_start_compound(self.agent_model)
        self.tyre_life      = 1
        self.safety_car_active = False
        self.compounds_used = {self.compound}
        self.pit_count      = 0
        self.total_time     = 0.0
        self.lap_times      = []
        self.sc_history     = []
        self._init_opponents()
        self.driver_history = {prof["Driver"]: []}
        for d in self.opponent_times:
            self.driver_history[d] = []

        return self._get_observation(), {
            "race":             race,
            "total_laps":       self.total_laps,
            "starting_position": self.position,
            "agent_driver":     prof["Driver"],
            "agent_team":       prof["Team"],
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        assert self.action_space.contains(action)
        self.safety_car_active = self.current_lap in self._sc_laps
        self.sc_history.append(self.safety_car_active)

        pit_this_lap = False
        extra_time   = 0.0

        if action != 0:
            self.compound = action - 1
            self.tyre_life = 1
            self.compounds_used.add(self.compound)
            self.pit_count  += 1
            pit_this_lap = True
            extra_time   = self.pit_time_loss * (SC_PIT_DISCOUNT if self.safety_car_active else 1.0)

        lap_time = self._predict_laptime(
            self.agent_model, self.compound, self.tyre_life,
            self.current_lap, self.position, extra_time)
        self.total_time += lap_time
        self.lap_times.append(lap_time)
        # registra histórico de lap times por piloto
        agent_driver = self.agent_model["profile"]["Driver"]
        self.driver_history.setdefault(agent_driver, []).append(lap_time)
        self._simulate_opponents()
        self._update_positions(agent_pit=pit_this_lap)

        if not pit_this_lap:
            self.tyre_life += 1

        finished_lap     = self.current_lap
        self.current_lap += 1
        terminated       = self.current_lap > self.total_laps
        reward           = self._compute_reward(terminated)

        return self._get_observation(), reward, terminated, False, {
            "lap":       finished_lap,
            "position":  self.position,
            "compound":  list(COMPOUND_MAIN.keys())[self.compound] if self.compound < 3 else "MEDIUM",
            "tyre_life": self.tyre_life,
            "lap_time":  lap_time,
            "pit":       pit_this_lap,
            "sc":        self.safety_car_active,
        }

    # ------------------------------------------------------------------
    # Motor neural de lap time
    # ------------------------------------------------------------------

    def _predict_laptime(self, model, compound, tyre_life, lap_number, position, extra_time=0.0):
        rng = self.np_random   # reutiliza o RNG do gymnasium — respeita o seed do treino
        lt  = self.lap_predictor.predict(
            compound_id     = compound,
            tyre_life       = tyre_life,
            lap_number      = lap_number,
            lap_progress    = lap_number / max(self.total_laps, 1),
            position        = position,
            safety_car      = int(self.safety_car_active),
            team_gap        = model["team_gap"],
            driver_racepace = model["profile"]["RacePace"],
            driver_tyremgmt = model["profile"]["TyreManagement"],
            track_factor    = self.track_factor,
            noise_sigma     = model["noise_sigma"],
            rng             = rng,
        )
        return float(lt + extra_time)

    def _simulate_opponents(self):
        for driver, opp in self.opponent_times.items():
            opp["pit_this_lap"] = False
            m = opp["model"]
            if self._should_pit(opp["tyre_life"], opp["compound"], self.current_lap, m, opp["compounds_used"]):
                nc = self._choose_next_compound(opp["compound"], self.current_lap, m, opp["compounds_used"])
                extra = self.pit_time_loss * (SC_PIT_DISCOUNT if self.safety_car_active else 1.0)
                opp["compound"]      = nc
                opp["tyre_life"]     = 1
                opp["pit_count"]    += 1
                opp["pit_this_lap"]  = True
                opp["compounds_used"].add(nc)
            else:
                extra = 0.0
            lt = self._predict_laptime(m, opp["compound"], opp["tyre_life"],
                                       self.current_lap, opp["position"], extra)
            opp["total_time"] += lt
            self.driver_history.setdefault(driver, []).append(lt)
            if not opp["pit_this_lap"]:
                opp["tyre_life"] += 1

    def _update_positions(self, agent_pit=False):
        standings = (
            [(self.agent_model["profile"]["Driver"], self.total_time)]
            + [(d, opp["total_time"]) for d, opp in self.opponent_times.items()]
        )
        standings.sort(key=lambda x: x[1])
        for i, (d, _) in enumerate(standings, 1):
            if d == self.agent_model["profile"]["Driver"]:
                self.position = i
            else:
                self.opponent_times[d]["position"] = i

    # ------------------------------------------------------------------
    # Observação (14 features)
    # ------------------------------------------------------------------

    def _gap_to_front(self):
        standings = (
            [(self.agent_model["profile"]["Driver"], self.total_time)]
            + [(d, opp["total_time"]) for d, opp in self.opponent_times.items()]
        )
        standings.sort(key=lambda x: x[1])
        for i, (d, t) in enumerate(standings):
            if d == self.agent_model["profile"]["Driver"]:
                return 0.0 if i == 0 else abs(t - standings[i - 1][1])
        return 0.0

    def _gap_to_leader(self):
        standings = sorted(
            [(self.agent_model["profile"]["Driver"], self.total_time)]
            + [(d, opp["total_time"]) for d, opp in self.opponent_times.items()],
            key=lambda x: x[1],
        )
        return max(0.0, self.total_time - standings[0][1])

    def _relative_pace(self):
        if not self.lap_times:
            return 0.5
        avg_agent = float(np.mean(self.lap_times[-5:]))
        opp_times = [opp["total_time"] for opp in self.opponent_times.values()]
        if not opp_times:
            return 0.5
        field_median = float(np.median(opp_times)) / max(self.current_lap, 1)
        return min(max((avg_agent - field_median) / 2.0 + 0.5, 0.0), 1.0)

    def _get_observation(self):
        n    = self.num_competitors
        prof = self.agent_model["profile"]
        return np.array([
            min(max(self.current_lap / max(self.total_laps, 1), 0.0), 1.0),
            min(max((n - self.position) / max(n - 1, 1), 0.0), 1.0),
            min(max(self.compound / 2.0, 0.0), 1.0),
            min(max(self.tyre_life / 50.0, 0.0), 1.0),
            min(max(self._gap_to_front() / 90.0, 0.0), 1.0),
            float(self.safety_car_active),
            min(max(self.pit_count / 5.0, 0.0), 1.0),
            min(max(len(self.compounds_used) / 3.0, 0.0), 1.0),
            self._relative_pace(),
            min(max(prof.get("AvgFinishPos_ML", 10.0) / 20.0, 0.0), 1.0),
            min(max(self.track_factor / 1.5, 0.0), 1.0),
            min(max((self.total_laps - self.current_lap) / max(self.total_laps, 1), 0.0), 1.0),
            min(max(self._expected_pit_stops / 4.0, 0.0), 1.0),
            min(max(self._gap_to_leader() / 120.0, 0.0), 1.0),
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Recompensa
    # ------------------------------------------------------------------

    def _compute_reward(self, terminated):
        n = self.num_competitors
        pos_reward = (n - self.position) / n * 0.5
        if not terminated:
            return float(pos_reward)
        final_reward    = (n - self.position) / max(n - 1, 1) * 20 - 10
        compound_penalty = -5.0 if len(self.compounds_used) < MANDATORY_COMPOUNDS else 0.0
        excess_pits     = max(0, self.pit_count - max(2, int(self._expected_pit_stops) + 1))
        pit_penalty     = excess_pits * -1.0
        avg_ml          = self.agent_model["profile"].get("AvgFinishPos_ML", 10.0)
        ml_bonus        = max(0.0, (avg_ml - self.position) * 0.3)
        return float(pos_reward + final_reward + compound_penalty + pit_penalty + ml_bonus)

    def render(self):
        if self.render_mode == "human":
            cname = list(COMPOUND_MAIN.keys())[self.compound] if self.compound < 3 else "?"
            print(f"Volta {self.current_lap:>3}/{self.total_laps} | "
                  f"P{self.position:>2} | {self.agent_model['profile']['Driver']} | "
                  f"Pneu: {cname:<6} ({self.tyre_life}v) | Pits: {self.pit_count}")

    def close(self):
        pass


# ============================================================================
# CALLBACKS
# ============================================================================

from stable_baselines3.common.callbacks import BaseCallback


class RobustCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="checkpoint", verbose=1):
        super().__init__(verbose)
        self.save_freq   = save_freq
        self.save_path   = Path(save_path)
        self.name_prefix = name_prefix
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = self.save_path / f"{self.name_prefix}_{self.n_calls}_steps"
            try:
                self.model.save(str(path))
                if self.verbose:
                    logger.info(f"✅ Checkpoint: {path}.zip ({self.n_calls} steps)")
            except Exception as e:
                logger.error(f"❌ Erro checkpoint: {e}")
        return True


class ProgressLogger(BaseCallback):
    def __init__(self, log_freq=5000, verbose=1):
        super().__init__(verbose)
        self.log_freq   = log_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0 and self.verbose:
            elapsed = time.time() - self.start_time
            sps     = self.n_calls / elapsed if elapsed > 0 else 0
            logger.info(f"📊 {self.n_calls} steps | {sps:.1f} steps/s | {elapsed/60:.1f}min")
        return True


# ============================================================================
# TREINAMENTO PRINCIPAL
# ============================================================================

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def run_training(
    timesteps: int = 500_000,
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    resume_from: Optional[str] = None,
    seed: int = 42,
    device: str = "auto",
    skip_laptime_training: bool = False,
):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        import torch
    except ImportError:
        logger.error("Instale: pip install stable-baselines3 torch scikit-learn")
        raise

    random.seed(seed)
    np.random.seed(seed)

    # ----------------------------------------------------------------
    # 1. Descoberta de dados
    # ----------------------------------------------------------------
    data_paths = discover_data_paths(data_root)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "training_output_v5"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir  = output_dir / "models"
    log_dir    = output_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    predictor_path = output_dir / "lap_time_predictor.pkl"

    # ----------------------------------------------------------------
    # 2. Carrega dados base
    # ----------------------------------------------------------------
    logger.info("📂 Carregando dados base...")
    for key in ["lap_data", "pit_stops", "race_events", "team_performances", "drivers"]:
        if not data_paths.get(key):
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {key}. Use --data-root.")

    lap_df       = pd.read_csv(data_paths["lap_data"])
    pit_df       = pd.read_csv(data_paths["pit_stops"])
    events_df    = pd.read_csv(data_paths["race_events"])
    team_perf_df = pd.read_csv(data_paths["team_performances"])
    drivers_df   = pd.read_csv(data_paths["drivers"])

    # Normaliza equipes
    team_perf_df["TeamKey"] = team_perf_df["Team"].astype(str).str.strip().map(
        lambda x: TEAM_NAME_MAP.get(x, x))
    drivers_df["Driver"]  = drivers_df["Driver"].astype(str).str.strip()
    drivers_df["TeamKey"] = drivers_df["Team"].astype(str).str.strip().map(
        lambda x: TEAM_NAME_MAP.get(x, x))

    ref = float(team_perf_df["AvgLapTime_s"].min())
    team_gap_map = {
        row["TeamKey"]: float(row["AvgLapTime_s"] - ref)
        for _, row in team_perf_df.iterrows()
    }

    # ----------------------------------------------------------------
    # 3. Track_Results e calibração
    # ----------------------------------------------------------------
    logger.info("📂 Carregando Track_Results...")
    track_results     = load_track_results(data_paths.get("track_results_dir"))
    track_calibration = build_track_calibration(track_results)
    logger.info(f"✅ {len(track_calibration)} pistas calibradas com dados reais 2025")

    # ----------------------------------------------------------------
    # 4. Datasets ML combinados
    # ----------------------------------------------------------------
    logger.info("📂 Carregando datasets ML...")
    combined_df    = load_and_merge_all_datasets(data_paths)
    driver_ml_stats = extract_driver_stats(combined_df)

    # Enriquece drivers_df com stats ML
    def get_ml_stat(d, k, default):
        return driver_ml_stats.get(d, {}).get(k, default)

    drivers_df["AvgFinishPosition_ML"] = drivers_df["Driver"].apply(
        lambda d: get_ml_stat(d, "avg_finish_position", 10.0))

    # ----------------------------------------------------------------
    # 5. Treino supervisionado do LapTimePredictor
    # ----------------------------------------------------------------
    if skip_laptime_training and predictor_path.exists():
        logger.info(f"📥 Carregando predictor existente: {predictor_path}")
        predictor = LapTimePredictor.load(predictor_path)
    else:
        logger.info("=" * 60)
        logger.info("🤖 FASE 1: TREINO SUPERVISIONADO DO LAP TIME PREDICTOR")
        logger.info("=" * 60)
        predictor = LapTimePredictor()
        metrics   = predictor.train(
            combined_df  = combined_df,
            lap_df       = lap_df,
            team_perf_df = team_perf_df,
            drivers_df   = drivers_df,
        )
        predictor.save(predictor_path)

        with open(output_dir / "predictor_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✅ Predictor salvo. Métricas: {metrics}")

    # ----------------------------------------------------------------
    # 6. Configura ambiente RL
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("🏎️  FASE 2: TREINAMENTO RL (PPO)")
    logger.info("=" * 60)

    def make_env():
        env = F1RaceEnvV5(
            data_paths        = data_paths,
            lap_predictor     = predictor,
            track_calibration = track_calibration,
            driver_ml_stats   = driver_ml_stats,
            team_gap_map      = team_gap_map,
            drivers_df        = drivers_df,
            lap_df            = lap_df,
            pit_df            = pit_df,
            events_df         = events_df,
            race_name         = None,
            render_mode       = None,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    # ----------------------------------------------------------------
    # 7. Cria ou carrega modelo PPO
    # ----------------------------------------------------------------
    if resume_from and Path(resume_from).exists():
        logger.info(f"📥 Resumindo de: {resume_from}")
        model = PPO.load(resume_from, env=vec_env, device=device)
    else:
        if torch.cuda.is_available():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        model = PPO(
            policy         = "MlpPolicy",
            env            = vec_env,
            learning_rate  = linear_schedule(3e-4),
            n_steps        = 2048,
            batch_size     = 128,
            n_epochs       = 20,
            gamma          = 0.999,
            gae_lambda     = 0.95,
            clip_range     = 0.2,
            ent_coef       = 0.08,
            vf_coef        = 0.5,
            max_grad_norm  = 0.5,
            tensorboard_log = str(log_dir),
            policy_kwargs  = dict(
                net_arch      = dict(pi=[256, 256], vf=[256, 256]),
                activation_fn = torch.nn.ReLU,
            ),
            verbose = 1,
            seed    = seed,
            device  = device,
        )
        logger.info(f"✅ Modelo PPO criado | obs={F1RaceEnvV5.OBS_DIM} | "
                    f"predictor={'neural' if predictor.is_trained else 'fallback'}")

    # ----------------------------------------------------------------
    # 8. Treina
    # ----------------------------------------------------------------
    config = {
        "version":             "v5_neural_laptime",
        "timesteps":           timesteps,
        "seed":                seed,
        "device":              str(device),
        "observation_dim":     F1RaceEnvV5.OBS_DIM,
        "predictor_trained":   predictor.is_trained,
        "track_calibrations":  len(track_calibration),
        "driver_ml_stats":     len(driver_ml_stats),
        "ml_rows":             len(combined_df),
        "data_root":           str(data_root),
        "created_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "training_config_v5.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("🚀 Iniciando PPO...")
    logger.info(f"   Timesteps  : {timesteps:,}")
    logger.info(f"   Motor      : {'GradientBoosting neural' if predictor.is_trained else 'fallback paramétrico'}")
    logger.info(f"   Obs space  : {F1RaceEnvV5.OBS_DIM} features")
    logger.info(f"   Pistas     : {len(track_calibration)} com dados reais 2025")
    logger.info(f"   Pilotos ML : {len(driver_ml_stats)}")

    try:
        model.learn(
            total_timesteps     = timesteps,
            callback            = [
                RobustCheckpointCallback(25_000, str(model_dir), "f1_ppo_v5"),
                ProgressLogger(5_000),
            ],
            progress_bar        = True,
            reset_num_timesteps = not bool(resume_from),
        )
        final = output_dir / f"f1_driver_final_v5_{timesteps}"
        model.save(str(final))
        logger.info("=" * 60)
        logger.info("✅ TREINAMENTO CONCLUÍDO!")
        logger.info(f"📦 {final}.zip")
        logger.info("=" * 60)
        return final

    except KeyboardInterrupt:
        path = output_dir / f"f1_interrupted_{int(time.time())}"
        model.save(str(path))
        logger.warning(f"⚠️  Interrompido. Salvo em {path}.zip")
        return path

    except Exception as e:
        logger.error(f"❌ {e}\n{traceback.format_exc()}")
        try:
            path = output_dir / f"f1_error_{int(time.time())}"
            model.save(str(path))
        except Exception:
            pass
        raise


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train V5 — Motor neural de lap time + RL PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python train_v5_neural_laptime.py --timesteps 500000
  python train_v5_neural_laptime.py --timesteps 500000 --data-root /path/RL_F1_for_performance
  python train_v5_neural_laptime.py --timesteps 500000 --skip-laptime-training
  python train_v5_neural_laptime.py --timesteps 500000 --resume models/checkpoint.zip
        """
    )
    p.add_argument("--timesteps",            type=int,  default=500_000)
    p.add_argument("--data-root",            type=str,  default="/home/otacs/Documentos/PI_IV",
                   help="Raiz do projeto (ex: /home/user/RL_F1_for_performance)")
    p.add_argument("--output-dir",           type=str,  default=None)
    p.add_argument("--resume",               type=str,  default=None)
    p.add_argument("--seed",                 type=int,  default=42)
    p.add_argument("--device",               type=str,  default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--skip-laptime-training", action="store_true",
                   help="Pula o treino do predictor e carrega um existente")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        run_training(
            timesteps             = args.timesteps,
            data_root             = Path(args.data_root) if args.data_root else None,
            output_dir            = Path(args.output_dir) if args.output_dir else None,
            resume_from           = args.resume,
            seed                  = args.seed,
            device                = args.device,
            skip_laptime_training = args.skip_laptime_training,
        )
    except Exception as e:
        logger.error(f"❌ Falha: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
