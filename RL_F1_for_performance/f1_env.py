import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
COMPOUND_NAMES = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
MANDATORY_COMPOUNDS = 2
PIT_STOP_TIME_LOSS = 23.5
GRID_GAP_SECONDS = 1.8
SC_PIT_DISCOUNT = 0.35
VSC_PIT_DISCOUNT = 0.65
MIN_PIT_GAP_LAPS = 12
MAX_AGENT_PIT_STOPS = 3
PACE_PIT_MIN_WEAR = 0.50
PACE_DROP_TO_PIT = 1.15
SC_GAP_MIN = 0.2
SC_GAP_MAX = 0.6
SC_PIT_POSITION_LOSS_MIN = 3
SC_PIT_POSITION_LOSS_MAX = 6

REAL_RACE_LAPS = {
    "Bahrain Grand Prix": 57,
    "Saudi Arabian Grand Prix": 50,
    "Australian Grand Prix": 58,
    "Japanese Grand Prix": 53,
    "Chinese Grand Prix": 56,
    "Miami Grand Prix": 57,
    "Emilia Romagna Grand Prix": 63,
    "Monaco Grand Prix": 78,
    "Canadian Grand Prix": 70,
    "Spanish Grand Prix": 66,
    "Austrian Grand Prix": 71,
    "British Grand Prix": 52,
    "Hungarian Grand Prix": 70,
    "Belgian Grand Prix": 44,
    "Dutch Grand Prix": 72,
    "Italian Grand Prix": 53,
    "Azerbaijan Grand Prix": 51,
    "Singapore Grand Prix": 62,
    "United States Grand Prix": 56,
    "Mexico City Grand Prix": 71,
    "São Paulo Grand Prix": 71,
    "Las Vegas Grand Prix": 50,
    "Qatar Grand Prix": 57,
    "Abu Dhabi Grand Prix": 58,
}
TRACK_PASSING_FACTOR = {
    "Monaco Grand Prix": 1.45,
    "Hungarian Grand Prix": 1.20,
    "Singapore Grand Prix": 1.18,
    "São Paulo Grand Prix": 1.00,
    "Austrian Grand Prix": 0.95,
    "Bahrain Grand Prix": 0.92,
}

TEAM_NAME_MAP = {
    "Red Bull": "Red Bull Racing",
    "Racing Bulls": "RB",
    "Sauber": "Kick Sauber",
    "Haas": "Haas F1 Team",
    "McLaren": "McLaren",
    "Mercedes": "Mercedes",
    "Ferrari": "Ferrari",
    "Aston Martin": "Aston Martin",
    "Alpine": "Alpine",
    "Williams": "Williams",
}

TEAM_COLORS = {
    "McLaren": "#ff8000",
    "Mercedes": "#27f4d2",
    "Red Bull Racing": "#0600ef",
    "Ferrari": "#dc0000",
    "RB": "#6692ff",
    "Kick Sauber": "#52e252",
    "Aston Martin": "#006f62",
    "Alpine": "#ff87bc",
    "Williams": "#64c4ff",
    "Haas F1 Team": "#ffffff",
}

DRIVER_CODES = {
    "Max Verstappen": "VER", "Lando Norris": "NOR", "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC",
    "George Russell": "RUS", "Oscar Piastri": "PIA", "Kimi Antonelli": "ANT", "Fernando Alonso": "ALO",
    "Carlos Sainz": "SAI", "Isack Hadjar": "HAD", "Gabriel Bortoleto": "BOR", "Oliver Bearman": "BEA",
    "Nico Hulkenberg": "HUL", "Alexander Albon": "ALB", "Pierre Gasly": "GAS", "Yuki Tsunoda": "TSU",
    "Esteban Ocon": "OCO", "Franco Colapinto": "COL", "Liam Lawson": "LAW", "Lance Stroll": "STR",
}


def _find_existing_file(filename, base_dir):
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "f1_data", os.path.basename(filename)),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "f1_data", os.path.basename(filename)),
    ]
    seen = []
    for path in candidates:
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.append(norm)
    for path in seen:
        if os.path.exists(path):
            return path
    return os.path.join(base_dir, filename)


class F1RaceEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path="f1_data",
        race_name=None,
        agent_team=None,
        agent_driver=None,
        team_perf_path=os.path.join("f1_data", "team_performances.csv"),
        drivers_path=os.path.join("f1_data", "drivers.csv"),
        starting_position=None,
        render_mode=None,
    ):
        super().__init__()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = data_path if os.path.isabs(data_path) else os.path.join(self.base_dir, data_path)
        self.team_perf_path = team_perf_path if os.path.isabs(team_perf_path) else _find_existing_file(team_perf_path, self.base_dir)
        self.drivers_path = drivers_path if os.path.isabs(drivers_path) else _find_existing_file(drivers_path, self.base_dir)
        self.race_name = race_name
        self.agent_team = agent_team
        self.agent_driver = agent_driver
        self.starting_position = starting_position
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.zeros(14, dtype=np.float32),
            high=np.ones(14, dtype=np.float32),
            dtype=np.float32,
        )

        self.current_lap = 0
        self.total_laps = 0
        self.position = 1
        self.compound = COMPOUND_MAP["MEDIUM"]
        self.tyre_life = 1
        self.safety_car_active = False
        self.vsc_active = False
        self.manual_sc_laps = set()
        self.manual_vsc_laps = set()
        self.compounds_used = set()
        self.pit_count = 0
        self.total_time = 0.0
        self.lap_times = []
        self.stint_lap_times = []
        self.last_pit_lap = -999
        self._sc_laps = set()
        self._vsc_laps = set()
        self.opponent_times = {}
        self.num_competitors = 20
        self.driver_history = {}
        self.sc_history = []
        self.grid_order = []
        self.track_factor = 1.0

        self._load_data()
        self._load_performance_data()

    def _load_data(self):
        self.lap_df = pd.read_csv(os.path.join(self.data_path, "lap_data.csv"))
        self.pit_df = pd.read_csv(os.path.join(self.data_path, "pit_stops.csv"))
        self.events_df = pd.read_csv(os.path.join(self.data_path, "race_events.csv"))
        self.available_races = sorted(self.lap_df["RaceName"].dropna().unique().tolist())
        self._precompute_tyre_model()

    def _load_performance_data(self):
        self.team_perf_df = pd.read_csv(self.team_perf_path)
        self.drivers_df = pd.read_csv(self.drivers_path)
        self.team_perf_df["TeamKey"] = self.team_perf_df["Team"].astype(str).str.strip().map(lambda x: TEAM_NAME_MAP.get(x, x))
        self.drivers_df["Driver"] = self.drivers_df["Driver"].astype(str).str.strip()
        self.drivers_df["TeamKey"] = self.drivers_df["Team"].astype(str).str.strip().map(lambda x: TEAM_NAME_MAP.get(x, x))
        ref = float(self.team_perf_df["AvgLapTime_s"].min())
        self.team_gap_map = {row["TeamKey"]: float(row["AvgLapTime_s"] - ref) for _, row in self.team_perf_df.iterrows()}

    def get_available_teams(self):
        return sorted(self.drivers_df["TeamKey"].dropna().unique().tolist())

    def get_drivers_by_team(self, team_name):
        team_key = self._canonical_team(team_name)
        return sorted(self.drivers_df[self.drivers_df["TeamKey"] == team_key]["Driver"].tolist())

    def get_available_races(self):
        return self.available_races

    def _precompute_tyre_model(self):
        self.tyre_model = {}
        base_defaults = {COMPOUND_MAP["SOFT"]: (0.0, 0.115), COMPOUND_MAP["MEDIUM"]: (0.35, 0.080), COMPOUND_MAP["HARD"]: (0.75, 0.058)}
        for compound, cid in COMPOUND_MAP.items():
            df = self.lap_df[self.lap_df["Compound"] == compound].copy()
            if df.empty:
                delta, deg = base_defaults[cid]
                self.tyre_model[cid] = {"compound_delta": delta, "deg_per_lap": deg}
                continue
            valid = df[["TyreLife", "LapTime_s"]].dropna()
            deg = max(float(np.polyfit(valid["TyreLife"], valid["LapTime_s"], 1)[0]), 0.0) if len(valid) > 5 else base_defaults[cid][1]
            self.tyre_model[cid] = {"compound_delta": base_defaults[cid][0], "deg_per_lap": max(deg, base_defaults[cid][1])}
        pit_valid = self.pit_df["PitDuration_s"].dropna()
        raw_pit = float(pit_valid.median()) if not pit_valid.empty else PIT_STOP_TIME_LOSS
        self.pit_time_loss = max(raw_pit, PIT_STOP_TIME_LOSS)

    def _canonical_team(self, team_name):
        return TEAM_NAME_MAP.get(team_name, team_name)

    def _get_team_base_time(self, team_name):
        team_name = self._canonical_team(team_name)
        row = self.team_perf_df[self.team_perf_df["TeamKey"] == team_name]
        if row.empty:
            return float(self.team_perf_df["AvgLapTime_s"].median())
        return float(row.iloc[0]["AvgLapTime_s"])

    def _get_driver_profile(self, driver_name=None, team_name=None):
        df = self.drivers_df.copy()
        if driver_name is not None:
            df = df[df["Driver"] == driver_name]
        if team_name is not None:
            df = df[df["TeamKey"] == self._canonical_team(team_name)]
        row = df.iloc[0] if not df.empty else self.drivers_df.iloc[0]
        return {
            "Driver": row["Driver"],
            "Team": row["TeamKey"],
            "RacePace": float(row["RacePace"]),
            "TyreManagement": float(row["TyreManagement"]),
            "Overall": float(row["Overall"]),
            "Code": DRIVER_CODES.get(row["Driver"], row["Driver"][:3].upper()),
        }

    def _build_driver_model(self, profile):
        team = profile["Team"]
        team_base = self._get_team_base_time(team)
        team_gap = self.team_gap_map.get(team, 1.0)
        overall_delta = (98.5 - profile["Overall"]) * 0.060
        racepace_delta = (99.0 - profile["RacePace"]) * 0.045
        tyre_factor = 1.0 + (100.0 - profile["TyreManagement"]) * 0.004
        top_team_bonus = 0.0
        if team_gap <= 0.15:
            top_team_bonus = float(self.np_random.normal(0.0, 0.08))
        noise_sigma = 0.055 if team_gap <= 0.15 else 0.035
        return {
            "team_base_time": team_base,
            "team_gap": team_gap,
            "driver_delta": overall_delta + racepace_delta + top_team_bonus,
            "deg_factor": tyre_factor,
            "noise_sigma": noise_sigma,
            "profile": profile,
        }

    def _build_safety_car_schedule(self, race_events):
        self._sc_laps = set()
        self._vsc_laps = set()

        if "StatusName" not in race_events.columns:
            return

        sc_events = race_events[race_events["StatusName"] == "SafetyCar"]
        vsc_events = race_events[race_events["StatusName"] == "VirtualSafetyCar"]

        for events, target_set, duration in [
            (sc_events, self._sc_laps, 3),
            (vsc_events, self._vsc_laps, 2),
        ]:
            if events.empty:
                continue
            num_events = len(events)
            for i in range(num_events):
                center_lap = int(self.total_laps * (i + 1) / (num_events + 1))
                for offset in range(duration):
                    lap = center_lap + offset
                    if 1 <= lap <= self.total_laps:
                        target_set.add(lap)

    def activate_safety_car(self, start_lap=None, duration=3):
        """Ativa Safety Car manualmente. Exemplo: env.activate_safety_car(18, 4)."""
        if start_lap is None:
            start_lap = self.current_lap
        for lap in range(int(start_lap), int(start_lap) + int(duration)):
            if 1 <= lap <= self.total_laps:
                self.manual_sc_laps.add(lap)
                self._sc_laps.add(lap)
        self.manual_vsc_laps.difference_update(self.manual_sc_laps)
        self._vsc_laps.difference_update(self._sc_laps)

    def activate_vsc(self, start_lap=None, duration=2):
        """Ativa Virtual Safety Car manualmente. Exemplo: env.activate_vsc(22, 2)."""
        if start_lap is None:
            start_lap = self.current_lap
        for lap in range(int(start_lap), int(start_lap) + int(duration)):
            if 1 <= lap <= self.total_laps and lap not in self._sc_laps:
                self.manual_vsc_laps.add(lap)
                self._vsc_laps.add(lap)

    def clear_manual_safety_events(self):
        """Remove SC/VSC manuais sem apagar eventos vindos do dataset."""
        self._sc_laps.difference_update(self.manual_sc_laps)
        self._vsc_laps.difference_update(self.manual_vsc_laps)
        self.manual_sc_laps.clear()
        self.manual_vsc_laps.clear()

    def _choose_start_compound(self, model):
        team_gap = model["team_gap"]
        if self.total_laps >= 55:
            return COMPOUND_MAP["MEDIUM"] if team_gap < 0.30 else COMPOUND_MAP["HARD"]
        return COMPOUND_MAP["SOFT"] if team_gap < 0.15 else COMPOUND_MAP["MEDIUM"]

    def _stint_target(self, compound, model):
        base = {COMPOUND_MAP["SOFT"]: 11, COMPOUND_MAP["MEDIUM"]: 19, COMPOUND_MAP["HARD"]: 28}[compound]
        return base + int((model["profile"]["TyreManagement"] - 85) * 0.30)

    def _choose_next_compound(self, current_compound, lap, model, used):
        laps_left = self.total_laps - lap
        if laps_left <= 12:
            return COMPOUND_MAP["SOFT"] if current_compound != COMPOUND_MAP["SOFT"] else COMPOUND_MAP["MEDIUM"]
        if len(used) < 2:
            for c in [COMPOUND_MAP["HARD"], COMPOUND_MAP["MEDIUM"], COMPOUND_MAP["SOFT"]]:
                if c != current_compound and c not in used:
                    return c
        return COMPOUND_MAP["MEDIUM"] if current_compound == COMPOUND_MAP["HARD"] else COMPOUND_MAP["HARD"]

    def _should_pit(self, tyre_life, compound, lap, model, used):
        target = self._stint_target(compound, model)
        laps_left = self.total_laps - lap + 1
        if tyre_life >= target:
            return True
        if compound == COMPOUND_MAP["SOFT"] and tyre_life >= max(9, target - 1) and laps_left > 9:
            return True
        if self.safety_car_active and tyre_life >= max(8, int(target * 0.70)) and laps_left > 7:
            return True
        if len(used) < 2 and laps_left <= target:
            return True
        return False

    def _build_competitor_pool(self):
        profiles = []
        for _, row in self.drivers_df.iterrows():
            profiles.append({
                "Driver": row["Driver"],
                "Team": row["TeamKey"],
                "RacePace": float(row["RacePace"]),
                "TyreManagement": float(row["TyreManagement"]),
                "Overall": float(row["Overall"]),
                "Code": DRIVER_CODES.get(row["Driver"], row["Driver"][:3].upper()),
            })
        return profiles

    def _driver_grid_score(self, profile):
        team_gap = self.team_gap_map.get(profile["Team"], 1.0)
        return team_gap + (99.0 - profile["Overall"]) * 0.030 + (99.0 - profile["RacePace"]) * 0.020 + float(self.np_random.normal(0, 0.020))

    def _create_starting_grid(self):
        pool = [p for p in self._build_competitor_pool() if p["Driver"] != self.agent_model["profile"]["Driver"]]
        for p in pool:
            p["grid_score"] = self._driver_grid_score(p)
        pool.sort(key=lambda p: p["grid_score"])
        selected = pool[: max(0, self.num_competitors - 1)]

        agent_default_score = self._driver_grid_score(self.agent_model["profile"])
        if self.starting_position is not None:
            insert_pos = max(1, min(int(self.starting_position), self.num_competitors))
        else:
            insert_pos = 1 + sum(1 for p in selected if p["grid_score"] < agent_default_score)
            insert_pos = max(1, min(insert_pos, self.num_competitors))

        grid = []
        idx = 0
        for pos in range(1, self.num_competitors + 1):
            if pos == insert_pos:
                grid.append((pos, self.agent_model["profile"], True))
            else:
                grid.append((pos, selected[idx], False))
                idx += 1
        self.grid_order = grid
        self.position = insert_pos

    def _starting_offset(self, pos):
        return (pos - 1) * GRID_GAP_SECONDS + float(self.np_random.normal(0, 0.08))

    def _init_opponents(self):
        self.opponent_times = {}
        self._create_starting_grid()
        for pos, profile, is_agent in self.grid_order:
            if is_agent:
                continue
            model = self._build_driver_model(profile)
            start_comp = self._choose_start_compound(model)
            self.opponent_times[profile["Driver"]] = {
                "total_time": self._starting_offset(pos),
                "model": model,
                "compound": start_comp,
                "tyre_life": 1,
                "pit_count": 0,
                "position": pos,
                "pit_this_lap": False,
                "compounds_used": {start_comp},
            }
        self.total_time = self._starting_offset(self.position)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and isinstance(options, dict):
            self.race_name = options.get("race_name", self.race_name)
            self.agent_team = options.get("agent_team", self.agent_team)
            self.agent_driver = options.get("agent_driver", self.agent_driver)
            self.starting_position = options.get("starting_position", self.starting_position)
        race = self.race_name if self.race_name is not None else self.np_random.choice(self.available_races)
        race_laps = self.lap_df[self.lap_df["RaceName"] == race]
        race_events = self.events_df[self.events_df["RaceName"] == race]
        dataset_laps = int(race_laps["LapNumber"].max())
        self.total_laps = REAL_RACE_LAPS.get(race, dataset_laps)
        self.num_competitors = max(int(race_laps["Driver"].nunique()), 2)
        self.track_factor = TRACK_PASSING_FACTOR.get(race, 1.0)
        self._build_safety_car_schedule(race_events)

        if self.agent_driver is not None:
            agent_profile = self._get_driver_profile(driver_name=self.agent_driver)
        elif self.agent_team is not None:
            agent_profile = self._get_driver_profile(team_name=self.agent_team)
        else:
            agent_profile = self._get_driver_profile()

        self.agent_model = self._build_driver_model(agent_profile)
        self.current_lap = 1
        self.compound = self._choose_start_compound(self.agent_model)
        self.tyre_life = 1
        self.safety_car_active = False
        self.vsc_active = False
        self.compounds_used = {self.compound}
        self.pit_count = 0
        self.total_time = 0.0
        self.lap_times = []
        self.stint_lap_times = []
        self.last_pit_lap = -999
        self.sc_history = []
        self._init_opponents()
        self.driver_history = {self.agent_model["profile"]["Driver"]: []}
        for d in self.opponent_times:
            self.driver_history[d] = []
        return self._get_observation(), {
            "race": race,
            "total_laps": self.total_laps,
            "starting_position": self.position,
            "agent_driver": self.agent_model["profile"]["Driver"],
            "agent_team": self.agent_model["profile"]["Team"],
            "agent_code": self.agent_model["profile"]["Code"],
        }

    def _lap_noise_multiplier(self, lap):
        if lap <= 3:
            return 0.10
        if lap <= 8:
            return 0.35
        return 1.0

    def _track_position_penalty(self, current_position):
        return max(0.0, (current_position - 1) * 0.040 * self.track_factor)

    def _pit_loss(self):
        if self.safety_car_active:
            return self.pit_time_loss * SC_PIT_DISCOUNT
        if self.vsc_active:
            return self.pit_time_loss * VSC_PIT_DISCOUNT
        return self.pit_time_loss

    def _compute_lap_time(self, model, compound, tyre_life, current_position, extra_time=0.0):
        tyre = self.tyre_model[compound]
        base = model["team_base_time"] + model["driver_delta"] + tyre["compound_delta"]
        degradation = tyre["deg_per_lap"] * tyre_life * model["deg_factor"]
        sc_penalty = 32.0 if self.safety_car_active else (13.0 if self.vsc_active else 0.0)
        noise = float(self.np_random.normal(0, model["noise_sigma"] * self._lap_noise_multiplier(self.current_lap)))
        traffic = self._track_position_penalty(current_position)
        return float(base + degradation + traffic + sc_penalty + extra_time + noise)

    def _simulate_opponents(self):
        for driver_name, opp in self.opponent_times.items():
            opp["pit_this_lap"] = False
            model = opp["model"]
            if self._should_pit(opp["tyre_life"], opp["compound"], self.current_lap, model, opp["compounds_used"]):
                new_comp = self._choose_next_compound(opp["compound"], self.current_lap, model, opp["compounds_used"])
                extra_time = self._pit_loss()
                opp["compound"] = new_comp
                opp["tyre_life"] = 1
                opp["pit_count"] += 1
                opp["pit_this_lap"] = True
                opp["compounds_used"].add(new_comp)
            else:
                extra_time = 0.0
            lap_time = self._compute_lap_time(model, opp["compound"], opp["tyre_life"], opp["position"], extra_time)
            opp["total_time"] += lap_time
            if not opp["pit_this_lap"]:
                opp["tyre_life"] += 1

    def _append_driver_history(self, agent_pit=False):
        lap = self.current_lap
        self.driver_history[self.agent_model["profile"]["Driver"]].append({
            "lap": lap,
            "position": self.position,
            "compound": COMPOUND_NAMES[self.compound],
            "pit": agent_pit,
            "team": self.agent_model["profile"]["Team"],
            "code": self.agent_model["profile"]["Code"],
        })

        for d, opp in self.opponent_times.items():
            self.driver_history[d].append({
                "lap": lap,
                "position": opp["position"],
                "compound": COMPOUND_NAMES[opp["compound"]],
                "pit": opp["pit_this_lap"],
                "team": opp["model"]["profile"]["Team"],
                "code": opp["model"]["profile"]["Code"],
            })

    def _get_current_track_order(self):
        agent_name = self.agent_model["profile"]["Driver"]
        order = [(self.position, agent_name)]
        for d, opp in self.opponent_times.items():
            order.append((opp["position"], d))
        order.sort(key=lambda x: x[0])
        return [driver for _, driver in order]

    def _get_driver_total_time(self, driver_name):
        if driver_name == self.agent_model["profile"]["Driver"]:
            return self.total_time
        return self.opponent_times[driver_name]["total_time"]

    def _set_driver_position(self, driver_name, position):
        if driver_name == self.agent_model["profile"]["Driver"]:
            self.position = position
        else:
            self.opponent_times[driver_name]["position"] = position

    def _recompress_sc_gaps_without_overtakes(self, ordered_drivers):
        """Junta o grid atrás do Safety Car sem trocar a ordem de pista."""
        if not ordered_drivers:
            return

        leader_time = min(self._get_driver_total_time(driver) for driver in ordered_drivers)
        current_time = leader_time

        for idx, driver in enumerate(ordered_drivers):
            if idx == 0:
                new_time = leader_time
            else:
                gap = float(self.np_random.uniform(SC_GAP_MIN, SC_GAP_MAX))
                current_time += gap
                new_time = current_time

            if driver == self.agent_model["profile"]["Driver"]:
                self.total_time = new_time
            else:
                self.opponent_times[driver]["total_time"] = new_time

    def _update_positions_safety_car_locked(self, agent_pit=False):
        """Atualiza posições durante SC sem ultrapassagens na pista.

        Regras:
        - sem pit stop: a ordem da pista fica exatamente congelada;
        - com pit stop: posição só muda por causa dos boxes;
        - a perda de posição no pit sob SC é limitada para evitar quedas irreais
          como P1 -> P20 quando o Safety Car acabou de entrar.
        """
        agent_name = self.agent_model["profile"]["Driver"]
        ordered = self._get_current_track_order()
        original_index = {driver: idx for idx, driver in enumerate(ordered)}

        pitted = set()
        if agent_pit:
            pitted.add(agent_name)

        for driver_name, opp in self.opponent_times.items():
            if opp.get("pit_this_lap", False):
                pitted.add(driver_name)

        if pitted:
            stable_order = [driver for driver in ordered if driver not in pitted]

            if not stable_order:
                ordered = sorted(pitted, key=lambda d: original_index[d])
            else:
                for driver in sorted(pitted, key=lambda d: original_index[d]):
                    loss = int(
                        self.np_random.integers(
                            SC_PIT_POSITION_LOSS_MIN,
                            SC_PIT_POSITION_LOSS_MAX + 1,
                        )
                    )

                    target_idx = original_index[driver] + loss

                    removed_before = sum(
                        1 for p in pitted
                        if original_index[p] < original_index[driver]
                    )
                    target_idx -= removed_before

                    target_idx = max(0, min(target_idx, len(stable_order)))
                    stable_order.insert(target_idx, driver)

                ordered = stable_order

        for pos, driver_name in enumerate(ordered, start=1):
            self._set_driver_position(driver_name, pos)

        self._recompress_sc_gaps_without_overtakes(ordered)
        self._append_driver_history(agent_pit=agent_pit)

    def _update_positions(self, agent_pit=False):
        standings = [(self.agent_model["profile"]["Driver"], self.total_time)] + [
            (d, opp["total_time"]) for d, opp in self.opponent_times.items()
        ]
        standings.sort(key=lambda x: x[1])

        for i, (driver_name, _) in enumerate(standings, start=1):
            self._set_driver_position(driver_name, i)

        self._append_driver_history(agent_pit=agent_pit)

    def _get_live_gaps(self):
        standings = [(self.agent_model["profile"]["Driver"], self.total_time)] + [(d, opp["total_time"] ) for d, opp in self.opponent_times.items()]
        standings.sort(key=lambda x: x[1])
        leader_time = standings[0][1]
        gaps = {}
        prev_time = None
        for pos, (driver, total_time) in enumerate(standings, start=1):
            gaps[driver] = {
                "position": pos,
                "gap_to_leader": max(0.0, float(total_time - leader_time)),
                "gap_to_front": 0.0 if prev_time is None else max(0.0, float(total_time - prev_time)),
            }
            prev_time = total_time
        return gaps

    def _agent_tyre_wear_ratio(self):
        return self.tyre_life / max(self._stint_target(self.compound, self.agent_model), 1)

    def _agent_recent_pace_drop(self):
        """Queda de ritmo dentro do stint atual.

        Compara as últimas 3 voltas do stint com as 3 primeiras.
        Retorna 0 se ainda não houver dados suficientes.
        """
        if len(self.stint_lap_times) < 7:
            return 0.0

        early_pace = float(np.mean(self.stint_lap_times[:3]))
        recent_pace = float(np.mean(self.stint_lap_times[-3:]))
        return max(0.0, recent_pace - early_pace)

    def _should_agent_pit_under_sc(self):
        """Prioriza pit na primeira volta útil de Safety Car.

        Evita o agente esperar várias voltas de SC para parar.
        """
        if not self.safety_car_active:
            return False

        if not self._can_agent_pit_now():
            return False

        # Evita pit duplo: se acabou de parar, não para de novo logo em seguida.
        # Isso corrige casos em que o agente troca para MEDIUM no SC e,
        # na volta seguinte, para de novo para HARD.
        if self.current_lap - self.last_pit_lap <= 3:
            return False

        laps_left = self.total_laps - self.current_lap
        tyre_wear = self._agent_tyre_wear_ratio()

        # Se ainda precisa usar outro composto, SC é a melhor janela.
        if len(self.compounds_used) < MANDATORY_COMPOUNDS and self.current_lap < self.total_laps - 2:
            return True

        # SC perto do fim: trocar para Soft costuma ser uma oportunidade forte.
        if laps_left <= 15 and self.compound != COMPOUND_MAP["SOFT"]:
            return True

        # Pneu já razoavelmente usado: aproveita o SC para reduzir perda.
        if tyre_wear >= 0.35 and laps_left > 5:
            return True

        # Terceira parada estratégica sob SC, se ainda houver corrida pela frente.
        if self.pit_count == 2 and laps_left <= 30 and laps_left >= 3 and self.compound != COMPOUND_MAP["SOFT"]:
            return True

        return False

    def _can_agent_pit_now(self):
        laps_left = self.total_laps - self.current_lap

        # Limite normal: até 2 paradas.
        # Exceção: terceira parada só é liberada em Safety Car no fim da corrida.
        if self.pit_count >= 2:
            late_sc_third_stop = (
                self.pit_count == 2
                and self.safety_car_active
                and laps_left <= 30
                and laps_left >= 2
            )
            if not late_sc_third_stop:
                return False

        # Evita pit stops muito próximos.
        # Exceção: SC final pode justificar uma parada extra.
        if self.current_lap - self.last_pit_lap < MIN_PIT_GAP_LAPS:
            late_sc_short_gap = (
                self.safety_car_active
                and laps_left <= 30
                and self.pit_count == 2
            )
            if not late_sc_short_gap:
                return False

        if self.current_lap >= self.total_laps - 2:
            return False

        return True

    def _must_agent_pit_for_rules(self):
        # Garante 2 compostos sem permitir múltiplas paradas desnecessárias.
        return (
            len(self.compounds_used) < MANDATORY_COMPOUNDS
            and self.pit_count == 0
            and self.current_lap >= int(self.total_laps * 0.68)
            and self.current_lap < self.total_laps - 3
        )

    def _should_agent_pit_by_pace(self):
        tyre_wear = self._agent_tyre_wear_ratio()
        pace_drop = self._agent_recent_pace_drop()
        return tyre_wear >= PACE_PIT_MIN_WEAR and pace_drop >= PACE_DROP_TO_PIT

    def step(self, action):
        assert self.action_space.contains(action), f"Ação inválida: {action}"

        self.safety_car_active = self.current_lap in self._sc_laps
        self.vsc_active = (self.current_lap in self._vsc_laps) and not self.safety_car_active
        self.sc_history.append(self.safety_car_active or self.vsc_active)

        pit_this_lap = False
        extra_time = 0.0

        # Estratégia do agente:
        # - PPO decide normalmente.
        # - Pit só é aceito se respeitar janela mínima e limite de paradas.
        # - Pit forçado só acontece por queda real de pace ou regra dos 2 compostos.
        should_force_pit = (
            self._can_agent_pit_now()
            and (
                self._should_agent_pit_under_sc()
                or self._should_agent_pit_by_pace()
                or self._must_agent_pit_for_rules()
            )
        )

        if should_force_pit and (action == 0 or action - 1 == self.compound):
            action = self._choose_next_compound(
                self.compound,
                self.current_lap,
                self.agent_model,
                self.compounds_used,
            ) + 1

        if action != 0:
            requested_compound = action - 1

            # Bloqueia pits ruins: mesmo composto ou pit fora da janela estratégica.
            # A função _can_agent_pit_now já trata a exceção de 3º pit em SC final.
            if requested_compound == self.compound or not self._can_agent_pit_now():
                action = 0

        if action != 0:
            self.compound = action - 1
            self.tyre_life = 1
            self.compounds_used.add(self.compound)
            self.pit_count += 1
            self.last_pit_lap = self.current_lap
            self.stint_lap_times = []
            pit_this_lap = True
            extra_time = self._pit_loss()
        lap_time = self._compute_lap_time(self.agent_model, self.compound, self.tyre_life, self.position, extra_time)
        self.total_time += lap_time
        self.lap_times.append(lap_time)
        self.stint_lap_times.append(lap_time)
        self._simulate_opponents()

        if self.safety_car_active:
            self._update_positions_safety_car_locked(agent_pit=pit_this_lap)
        else:
            self._update_positions(agent_pit=pit_this_lap)
        if not pit_this_lap:
            self.tyre_life += 1
        finished_lap = self.current_lap
        self.current_lap += 1
        terminated = self.current_lap > self.total_laps
        reward = self._compute_reward(terminated)
        return self._get_observation(), reward, terminated, False, {
            "lap": finished_lap,
            "position": self.position,
            "compound": COMPOUND_NAMES[self.compound],
            "tyre_life": self.tyre_life,
            "lap_time": lap_time,
            "pit_this_lap": pit_this_lap,
            "safety_car": self.safety_car_active,
            "vsc": self.vsc_active,
            "compounds_used": len(self.compounds_used),
            "agent_driver": self.agent_model["profile"]["Driver"],
            "driver_history": self.driver_history,
            "agent_team": self.agent_model["profile"]["Team"],
            "agent_code": self.agent_model["profile"]["Code"],
            "sc_history": self.sc_history,
            "live_gaps": self._get_live_gaps(),
        }

    def _estimate_gap(self):
        standings = [(self.agent_model["profile"]["Driver"], self.total_time)] + [(d, opp["total_time"]) for d, opp in self.opponent_times.items()]
        standings.sort(key=lambda x: x[1])
        for idx, (d, t) in enumerate(standings):
            if d == self.agent_model["profile"]["Driver"]:
                return 0.0 if idx == 0 else abs(t - standings[idx - 1][1])
        return 0.0
    def _gap_to_front(self):
        standings = [(self.agent_model["profile"]["Driver"], self.total_time)] + [
        (d, opp["total_time"]) for d, opp in self.opponent_times.items()
         ]
        standings.sort(key=lambda x: x[1])

        for i, (driver, time) in enumerate(standings):
            if driver == self.agent_model["profile"]["Driver"]:
               return 0.0 if i == 0 else max(0.0, time - standings[i - 1][1])
        return 0.0


    def _gap_to_leader(self):
        standings = [(self.agent_model["profile"]["Driver"], self.total_time)] + [
        (d, opp["total_time"]) for d, opp in self.opponent_times.items()
        ]
        standings.sort(key=lambda x: x[1])
        return max(0.0, self.total_time - standings[0][1])


    def _relative_pace(self):
        if not self.lap_times:
          return 0.5

        avg_agent = float(np.mean(self.lap_times[-5:]))

        opp_avg = []
        for opp in self.opponent_times.values():
           if self.current_lap > 0:
               opp_avg.append(opp["total_time"] / max(self.current_lap, 1))

        if not opp_avg:
            return 0.5

        field_median = float(np.median(opp_avg))
        return min(max((avg_agent - field_median) / 2.0 + 0.5, 0.0), 1.0)


    def _get_observation(self):
        n = self.num_competitors
        prof = self.agent_model["profile"]

        expected_pit_stops = 2 if self.total_laps >= 50 else 1

        avg_finish_ml = prof.get("AvgFinishPos_ML", self.position)

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
            min(max(avg_finish_ml / 20.0, 0.0), 1.0),
            min(max(self.track_factor / 1.5, 0.0), 1.0),
            min(max((self.total_laps - self.current_lap) / max(self.total_laps, 1), 0.0), 1.0),
            min(max(expected_pit_stops / 4.0, 0.0), 1.0),
            min(max(self._gap_to_leader() / 120.0, 0.0), 1.0),
        ], dtype=np.float32)

    def _compute_reward(self, terminated):
        n = self.num_competitors
        position_reward = (n - self.position) / n * 0.5

        tyre_wear = self._agent_tyre_wear_ratio()
        pace_drop = self._agent_recent_pace_drop()
        tyre_penalty = -1.5 if tyre_wear >= PACE_PIT_MIN_WEAR and pace_drop >= PACE_DROP_TO_PIT and self.pit_count == 0 else 0.0

        if not terminated:
            return float(position_reward + tyre_penalty)

        final_reward = (n - self.position) / max(n - 1, 1) * 20 - 10
        compound_penalty = -25.0 if len(self.compounds_used) < MANDATORY_COMPOUNDS else 0.0
        no_pit_penalty = -80.0 if self.pit_count == 0 else 0.0
        pit_penalty = max(0, self.pit_count - 3) * -1.0
        return float(position_reward + final_reward + compound_penalty + no_pit_penalty + pit_penalty)

    def render(self):
        if self.render_mode == "human":
            print(f"Volta {self.current_lap:>3}/{self.total_laps} | P{self.position:>2} | {self.agent_model['profile']['Driver']} | Pneu: {COMPOUND_NAMES[self.compound]:<6} ({self.tyre_life} voltas) | Pit stops: {self.pit_count}")

    def close(self):
        pass