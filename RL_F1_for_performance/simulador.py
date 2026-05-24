import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from stable_baselines3 import PPO
import numpy as np
import sys
sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.numeric"] = np.core.numeric

from f1_env import F1RaceEnv, TEAM_COLORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "f1_data")
OUT_DIR = os.path.join(BASE_DIR, "output_media")
os.makedirs(OUT_DIR, exist_ok=True)


def choose_race_setup_gui_or_terminal():
    env_preview = F1RaceEnv(
        data_path=DATA_DIR,
        team_perf_path=os.path.join("f1_data", "team_performances.csv"),
        drivers_path=os.path.join("f1_data", "drivers.csv"),
        render_mode=None,
    )
    teams = env_preview.get_available_teams()
    races = env_preview.get_available_races()

    try:
        import tkinter as tk
        from tkinter import ttk

        default_team = teams[0]
        default_driver = env_preview.get_drivers_by_team(default_team)[0]
        result = {"team": default_team, "driver": default_driver, "race": races[0], "grid_pos": "AUTO"}

        root = tk.Tk()
        root.title("F1 Race Setup")
        root.geometry("500x320")
        root.resizable(False, False)
        root.configure(bg="#0a0d1a")

        title = tk.Label(root, text="Escolha circuito, equipe, piloto e grid", font=("Arial", 16, "bold"), fg="white", bg="#0a0d1a")
        title.pack(pady=16)

        frame = tk.Frame(root, bg="#0a0d1a")
        frame.pack(pady=8)

        tk.Label(frame, text="Circuito", fg="white", bg="#0a0d1a", font=("Arial", 11)).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        race_var = tk.StringVar(value=races[0])
        race_combo = ttk.Combobox(frame, textvariable=race_var, values=races, state="readonly", width=32)
        race_combo.grid(row=0, column=1, padx=10, pady=8)

        tk.Label(frame, text="Equipe", fg="white", bg="#0a0d1a", font=("Arial", 11)).grid(row=1, column=0, padx=10, pady=8, sticky="w")
        team_var = tk.StringVar(value=default_team)
        team_combo = ttk.Combobox(frame, textvariable=team_var, values=teams, state="readonly", width=32)
        team_combo.grid(row=1, column=1, padx=10, pady=8)

        tk.Label(frame, text="Piloto", fg="white", bg="#0a0d1a", font=("Arial", 11)).grid(row=2, column=0, padx=10, pady=8, sticky="w")
        initial_drivers = env_preview.get_drivers_by_team(default_team)
        driver_var = tk.StringVar(value=initial_drivers[0])
        driver_combo = ttk.Combobox(frame, textvariable=driver_var, values=initial_drivers, state="readonly", width=32)
        driver_combo.grid(row=2, column=1, padx=10, pady=8)

        tk.Label(frame, text="Posição inicial", fg="white", bg="#0a0d1a", font=("Arial", 11)).grid(row=3, column=0, padx=10, pady=8, sticky="w")
        grid_options = ["AUTO"] + [str(i) for i in range(1, 21)]
        grid_var = tk.StringVar(value="AUTO")
        grid_combo = ttk.Combobox(frame, textvariable=grid_var, values=grid_options, state="readonly", width=32)
        grid_combo.grid(row=3, column=1, padx=10, pady=8)

        hint = tk.Label(root, text="AUTO = equipes mais fiéis, gaps mais abertos e pit com perda real", font=("Arial", 9), fg="#cfd3db", bg="#0a0d1a")
        hint.pack(pady=4)

        def update_drivers(event=None):
            selected_team = team_var.get()
            drivers = env_preview.get_drivers_by_team(selected_team)
            driver_combo["values"] = drivers
            if drivers:
                driver_var.set(drivers[0])

        def confirm():
            result["team"] = team_var.get()
            result["driver"] = driver_var.get()
            result["race"] = race_var.get()
            result["grid_pos"] = grid_var.get()
            root.destroy()

        team_combo.bind("<<ComboboxSelected>>", update_drivers)

        btn = tk.Button(root, text="Iniciar corrida", command=confirm, bg="#ff8000", fg="black", font=("Arial", 12, "bold"), relief="flat", padx=12, pady=8)
        btn.pack(pady=18)

        root.mainloop()
        return result

    except Exception:
        print("\n=== Escolha do circuito ===")
        for i, race in enumerate(races, start=1):
            print(f"{i}. {race}")
        race_idx = int(input("Escolha o circuito: ")) - 1
        race = races[max(0, min(race_idx, len(races) - 1))]

        print("\n=== Escolha da equipe ===")
        for i, team in enumerate(teams, start=1):
            print(f"{i}. {team}")
        team_idx = int(input("Escolha a equipe: ")) - 1
        team = teams[max(0, min(team_idx, len(teams) - 1))]

        drivers = env_preview.get_drivers_by_team(team)
        print("\n=== Escolha do piloto ===")
        for i, driver in enumerate(drivers, start=1):
            print(f"{i}. {driver}")
        driver_idx = int(input("Escolha o piloto: ")) - 1
        driver = drivers[max(0, min(driver_idx, len(drivers) - 1))]

        grid_raw = input("Posição inicial do agente (1-20) ou ENTER para AUTO: ").strip()
        grid_pos = "AUTO" if grid_raw == "" else grid_raw
        return {"team": team, "driver": driver, "race": race, "grid_pos": grid_pos}


def run_selected_race(agent_team, agent_driver, race_name, starting_position=None, interval=100, save_gif=False, save_mp4=False, gif_name="lap_chart.gif", mp4_name="lap_chart.mp4"):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Button

    env = F1RaceEnv(
        data_path=DATA_DIR,
        team_perf_path=os.path.join("f1_data", "team_performances.csv"),
        drivers_path=os.path.join("f1_data", "drivers.csv"),
        race_name=race_name,
        agent_team=agent_team,
        agent_driver=agent_driver,
        starting_position=starting_position,
        render_mode=None,
    )

    model = PPO.load(
        os.path.join(
            BASE_DIR,
            "..",
            "training_session_algorithm",
            "training_output_v5",
            "f1_driver_final_v5_500000",
        ),
        device="cpu",
        custom_objects={
            "learning_rate": 0.0003,
            "lr_schedule": lambda _: 0.0003,
            "clip_range": lambda _: 0.2,
            "seed": 0,
            "_last_obs": None,
            "_last_episode_starts": None,
            "_last_original_obs": None,
            "ep_info_buffer": None,
            "ep_success_buffer": None,
            "_vec_normalize_env": None,
        },
    )

    obs, info = env.reset(seed=123)

    total_reward = 0.0
    final_report_shown = False
    last_step_info = None

    fig = plt.figure(figsize=(18, 9), facecolor="#070b18")
    ax = fig.add_axes([0.05, 0.11, 0.68, 0.76])
    header_ax = fig.add_axes([0.04, 0.89, 0.92, 0.08])
    right_ax = fig.add_axes([0.75, 0.11, 0.22, 0.76])
    button_sc_ax = fig.add_axes([0.76, 0.005, 0.09, 0.035])
    button_vsc_ax = fig.add_axes([0.87, 0.005, 0.09, 0.035])

    btn_sc = Button(button_sc_ax, "SC")
    btn_vsc = Button(button_vsc_ax, "VSC")

    def trigger_sc(event):
       env.activate_safety_car(start_lap=env.current_lap, duration=3)
       print(f"Safety Car ativado na volta {env.current_lap}")

    def trigger_vsc(event):
       env.activate_vsc(start_lap=env.current_lap, duration=2)
       print(f"VSC ativado na volta {env.current_lap}")

    btn_sc.on_clicked(trigger_sc)
    btn_vsc.on_clicked(trigger_vsc)

    compound_short = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}
    compound_fill = {"SOFT": "#ff2d2d", "MEDIUM": "#ffd400", "HARD": "#f3f3f3"}

    terminated = False
    truncated = False

    obs_list = [float(x) for x in np.asarray(obs).flatten().tolist()]
    obs_tensor = torch.tensor([obs_list], dtype=torch.float32, device=model.device)

    with torch.no_grad():
        action_tensor = model.policy.get_distribution(obs_tensor).get_actions(deterministic=True)

    action = int(action_tensor.item())
   

    def ordinal(n):
        if n == 1:
            return "1st"
        if n == 2:
            return "2nd"
        if n == 3:
            return "3rd"
        return f"{n}th"

    def fmt_gap(x):
        return "LEADER" if x <= 1e-9 else f"+{x:.3f}s"

    def short_name(name, max_len=12):
        parts = [p for p in name.split() if p]
        if not parts:
            return name
        surname = parts[-1]
        if len(surname) <= max_len:
            return surname
        return surname[:max_len-1] + "…"

    def show_final_report(step_info):
        import tkinter as tk
        from tkinter import messagebox

        if step_info is None:
            print("Relatório final não disponível: step_info vazio.")
            return

        agent = step_info["agent_driver"]
        history = step_info["driver_history"].get(agent, [])

        if not history:
            print("Relatório final não disponível: histórico do agente vazio.")
            return

        start_pos = int(info["starting_position"])
        final_pos = int(history[-1]["position"])
        gained_positions = start_pos - final_pos

        stints = []
        current_compound = history[0]["compound"]
        stint_start_lap = history[0]["lap"]
        stint_laps = 0

        for lap_data in history:
            compound = lap_data["compound"]

            if compound != current_compound:
                stint_end_lap = lap_data["lap"] - 1
                stints.append({
                    "compound": current_compound,
                    "start": stint_start_lap,
                    "end": stint_end_lap,
                    "laps": stint_laps,
                })

                current_compound = compound
                stint_start_lap = lap_data["lap"]
                stint_laps = 1
            else:
                stint_laps += 1

        stints.append({
            "compound": current_compound,
            "start": stint_start_lap,
            "end": history[-1]["lap"],
            "laps": stint_laps,
        })

        total_laps_completed = history[-1]["lap"]
        sc_vsc_laps = sum(1 for value in step_info.get("sc_history", []) if value)
        total_pits = sum(1 for lap_data in history if lap_data.get("pit", False))

        report = []
        report.append("RELATÓRIO FINAL DA CORRIDA")
        report.append("")
        report.append(f"Piloto: {agent}")
        report.append(f"Equipe: {step_info['agent_team']}")
        report.append(f"Circuito: {race_name}")
        report.append("")
        report.append(f"Posição de largada: P{start_pos}")
        report.append(f"Posição final: P{final_pos}")
        report.append(f"Posições ganhas/perdidas: {gained_positions:+d}")
        report.append("")
        report.append(f"Voltas completadas: {total_laps_completed}/{env.total_laps}")
        report.append(f"Pit Stops: {total_pits}")
        report.append(f"Voltas sob SC/VSC: {sc_vsc_laps}")
        report.append(f"Reward acumulado: {total_reward:.2f}")
        report.append("")
        report.append("STINTS:")

        for i, stint in enumerate(stints, start=1):
            report.append(
                f"{i}. {stint['compound']}: voltas {stint['start']}-{stint['end']} "
                f"({stint['laps']} voltas)"
            )

        report_text = "\n".join(report)

        print("\n" + "=" * 70)
        print(report_text)
        print("=" * 70)

        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("Relatório Final da Corrida", report_text)
            root.destroy()
        except Exception as e:
            print(f"Não foi possível abrir a janela do relatório: {e}")

    def draw_frame(_):
        nonlocal obs, terminated, truncated
        nonlocal total_reward, final_report_shown, last_step_info
        if not (terminated or truncated):
            obs_array = np.asarray(obs, dtype=np.float32)

            obs_list = [float(x) for x in np.asarray(obs).flatten().tolist()]
            obs_tensor = torch.tensor([obs_list], dtype=torch.float32, device=model.device)

            with torch.no_grad():
                  action_tensor = model.policy.get_distribution(obs_tensor).get_actions(deterministic=True)

            action = int(action_tensor.item())
            if hasattr(action, "__len__"):
               action = int(action[0])
            else:
               action = int(action)

            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += float(reward)
            last_step_info = step_info
        
        else:
            step_info = {
                "lap": env.total_laps,
                "driver_history": env.driver_history,
                "agent_driver": env.agent_model["profile"]["Driver"],
                "agent_team": env.agent_model["profile"]["Team"],
                "agent_code": env.agent_model["profile"]["Code"],
                "sc_history": env.sc_history,
                "live_gaps": env._get_live_gaps(),
            }

        history = step_info["driver_history"]
        sc_history = step_info.get("sc_history", [])
        live_gaps = step_info.get("live_gaps", {})

        ax.clear(); header_ax.clear(); right_ax.clear()
        ax.set_facecolor("#0a0d1a"); header_ax.set_facecolor("#0a0d1a"); right_ax.set_facecolor("#0a0d1a")
        header_ax.set_axis_off(); right_ax.set_axis_off()

        for lap_idx, is_sc in enumerate(sc_history, start=1):
            if is_sc:
                ax.axvspan(lap_idx - 0.5, lap_idx + 0.5, color="#ffd400", alpha=0.10, zorder=0)

        current_positions = []
        for driver, d_hist in history.items():
            if not d_hist:
                continue
            laps = [p["lap"] for p in d_hist]
            positions = [p["position"] for p in d_hist]
            team = d_hist[-1]["team"]
            color = TEAM_COLORS.get(team, "#d0d0d0")
            code = d_hist[-1].get("code", driver[:3].upper())
            is_agent = driver == step_info["agent_driver"]
            lw = 3.4 if is_agent else 2.0
            alpha = 1.0 if is_agent else 0.92
            z = 8 if is_agent else 3

            for i in range(1, len(laps)):
                ax.plot([laps[i - 1], laps[i]], [positions[i - 1], positions[i]], color=color, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=z)

            for pt in d_hist:
                if pt.get("pit", False):
                    ax.scatter(pt["lap"], pt["position"], s=120 if is_agent else 65, marker="D", color=compound_fill.get(pt["compound"], color), edgecolors="white", linewidths=1.4, zorder=10)
                    ax.annotate(f"P {compound_short.get(pt['compound'], pt['compound'][0])}", (pt["lap"], pt["position"]), textcoords="offset points", xytext=(0, -14), ha="center", color="white", fontsize=7 if not is_agent else 8, fontweight="bold", zorder=11)

            stint_start = 0
            for i in range(1, len(d_hist)):
                if d_hist[i]["compound"] != d_hist[i - 1]["compound"]:
                    start_pt = d_hist[stint_start]
                    ax.text(start_pt["lap"], start_pt["position"] - 0.25, compound_short.get(start_pt["compound"], start_pt["compound"][0]), color=compound_fill.get(start_pt["compound"], color), fontsize=7, fontweight="bold", ha="center", va="bottom", zorder=9)
                    stint_start = i
            start_pt = d_hist[stint_start]
            ax.text(start_pt["lap"], start_pt["position"] - 0.25, compound_short.get(start_pt["compound"], start_pt["compound"][0]), color=compound_fill.get(start_pt["compound"], color), fontsize=7, fontweight="bold", ha="center", va="bottom", zorder=9)
            gap_info = live_gaps.get(driver, {})
            gap_to_leader = float(gap_info.get("gap_to_leader", 0.0))
            gap_to_front = float(gap_info.get("gap_to_front", 0.0))
            current_positions.append((positions[-1], driver, code, color, is_agent, gap_to_leader, gap_to_front))

        ax.set_xlim(1, env.total_laps)
        ax.set_ylim(env.num_competitors + 0.5, 0.5)
        ax.set_xticks(range(1, env.total_laps + 1, max(1, env.total_laps // 10)))
        ax.set_yticks(range(1, env.num_competitors + 1))
        ax.tick_params(axis="x", colors="white", labelsize=10)
        ax.tick_params(axis="y", colors="white", labelsize=10)
        ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
        for spine in ax.spines.values():
            spine.set_color("#293043"); spine.set_linewidth(1.0)
        ax.grid(True, axis="y", color="white", alpha=0.08, linestyle="-")
        ax.grid(False, axis="x")

        header_ax.text(0.01, 0.60, "F1® RACE LAP CHART", color="white", fontsize=22, fontweight="bold")
        header_ax.text(0.29, 0.60, f"LAP: {step_info['lap']}/{env.total_laps}", color="white", fontsize=20, fontweight="bold")
        header_ax.text(0.55, 0.60, race_name.replace(' Grand Prix', '').upper(), color="white", fontsize=18)
        header_ax.text(0.99, 0.60, f"GRID P{info['starting_position']}", color="#ff3b30", fontsize=18, fontweight="bold", ha="right")
        header_ax.add_patch(Rectangle((0.0, 0.05), 1.0, 0.02, color="#1e2438", alpha=1.0))

        current_positions.sort(key=lambda x: x[0])
        right_ax.set_xlim(0, 1); right_ax.set_ylim(env.num_competitors + 0.5, 0.5)
        right_ax.text(0.80, 0.15, "LÍDER", color="#8f9bb3", fontsize=7.5, ha="right")
        right_ax.text(0.98, 0.15, "FRENTE", color="#8f9bb3", fontsize=7.5, ha="right")
        right_ax.plot([0.84, 0.84], [0.4, env.num_competitors + 0.3], color="#243047", linewidth=0.8, alpha=0.9)

        for pos, driver, code, color, is_agent, gap_leader, gap_front in current_positions:
            display_name = short_name(driver.upper(), max_len=10)
            right_ax.scatter(0.04, pos, s=28 if is_agent else 18, color=color, zorder=10)
            right_ax.text(0.09, pos, ordinal(int(pos)), color="white", fontsize=10.0, va="center")
            right_ax.text(0.24, pos, display_name, color="#ffffff" if is_agent else "#cfd3db", fontsize=9.0, va="center", fontweight="bold" if is_agent else "normal")
            right_ax.text(0.80, pos, fmt_gap(float(gap_leader)), color="#ffd166" if gap_leader > 0 else "#ffffff", fontsize=7.8, va="center", ha="right")
            right_ax.text(0.98, pos, "-" if pos == 1 else f"+{float(gap_front):.3f}", color="#9ad1ff", fontsize=7.8, va="center", ha="right")

        fig.texts.clear()

        fig.text(
            0.05,
            0.04,
            f"AGENT: {step_info['agent_driver']} ({step_info['agent_team']})",
            color="#d7d7d7",
            fontsize=10,
        )

        tyre_target = env._stint_target(env.compound, env.agent_model)
        tyre_wear_pct = float(min(100, (env.tyre_life / max(tyre_target, 1)) * 100))

        fig.text(
            0.05,
            0.015,
            f"DESGASTE PNEU: {tyre_wear_pct:.1f}%",
            color="#ffd400" if tyre_wear_pct >= 60 else "#d7d7d7",
            fontsize=10,
            fontweight="bold",
       )
        fig.text(0.33, 0.04, "SOFT=S   MEDIUM=M   HARD=H   PIT=◆   SC=faixa amarela   GAPS=líder e carro da frente", color="#d7d7d7", fontsize=10)

        if terminated and not final_report_shown and step_info["lap"] >= env.total_laps:
            final_report_shown = True
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            show_final_report(last_step_info)

        return []

    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=range(env.total_laps + 3),
        interval=interval,
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )

    if save_gif:
        gif_path = os.path.join(OUT_DIR, gif_name)
        ani.save(gif_path, writer=animation.PillowWriter(fps=max(1, int(1000 / interval))))
        print(f"GIF salvo em: {gif_path}")
    if save_mp4:
        mp4_path = os.path.join(OUT_DIR, mp4_name)
        ani.save(mp4_path, writer=animation.FFMpegWriter(fps=max(1, int(1000 / interval))))
        print(f"MP4 salvo em: {mp4_path}")

    plt.show()
    return ani


if __name__ == "__main__":
    setup = choose_race_setup_gui_or_terminal()
    grid_value = None if str(setup["grid_pos"]).upper() == "AUTO" else int(setup["grid_pos"])
    run_selected_race(
        agent_team=setup["team"],
        agent_driver=setup["driver"],
        race_name=setup["race"],
        starting_position=grid_value,
        interval=120,
        save_gif=False,
        save_mp4=False,
    )
