import sys
import os
import matplotlib

# --- CONFIGURAÇÃO DE BACKEND E INTERFACE ---
try:
    import tkinter as tk
    from tkinter import ttk
    # Se o tkinter existe, tentamos usar o backend interativo
    if os.environ.get('DISPLAY', '') == '':
        print("[Aviso] Variável DISPLAY vazia. Usando backend 'Agg' (sem janela).")
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')
except ImportError:
    print("[Aviso] Tkinter não encontrado. Usando modo terminal e backend 'Agg'.")
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# --- CONFIGURAÇÃO DE DIRETÓRIOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    from f1_env import F1RaceEnv, TEAM_COLORS
except ImportError:
    print("Erro: f1_env.py não encontrado. Certifique-se que o arquivo está na mesma pasta.")
    sys.exit(1)

DATA_DIR = os.path.join(BASE_DIR, "f1_data")
OUT_DIR = os.path.join(BASE_DIR, "output_media")
os.makedirs(OUT_DIR, exist_ok=True)

# Fonte segura para Linux (DejaVu Sans é padrão na maioria das distros)
FONT_NAME = "DejaVu Sans" if os.name != 'nt' else "Arial"

def choose_race_setup():
    """Interface de configuração com fallback para terminal."""
    env_preview = F1RaceEnv(
        data_path=DATA_DIR,
        team_perf_path=os.path.join(DATA_DIR, "team_performances.csv"),
        drivers_path=os.path.join(DATA_DIR, "drivers.csv"),
        render_mode=None,
    )
    teams = env_preview.get_available_teams()
    races = env_preview.get_available_races()

    try:
        import tkinter as tk # Re-import local para garantir
        root = tk.Tk()
        root.title("F1 Setup - Linux Edition")
        root.geometry("520x380")
        root.configure(bg="#0a0d1a")

        result = {"team": teams[0], "driver": "", "race": races[0], "grid_pos": "AUTO"}

        tk.Label(root, text="F1 RACE SIMULATOR", font=(FONT_NAME, 16, "bold"), fg="#ff8000", bg="#0a0d1a").pack(pady=15)
        
        frame = tk.Frame(root, bg="#0a0d1a")
        frame.pack(pady=10, padx=20)

        # Combos
        tk.Label(frame, text="Circuito:", fg="white", bg="#0a0d1a").grid(row=0, column=0, sticky="w", pady=5)
        race_cb = ttk.Combobox(frame, values=races, width=30, state="readonly")
        race_cb.current(0); race_cb.grid(row=0, column=1, pady=5)

        tk.Label(frame, text="Equipe:", fg="white", bg="#0a0d1a").grid(row=1, column=0, sticky="w", pady=5)
        team_cb = ttk.Combobox(frame, values=teams, width=30, state="readonly")
        team_cb.current(0); team_cb.grid(row=1, column=1, pady=5)

        tk.Label(frame, text="Piloto:", fg="white", bg="#0a0d1a").grid(row=2, column=0, sticky="w", pady=5)
        driver_cb = ttk.Combobox(frame, width=30, state="readonly")
        driver_cb.grid(row=2, column=1, pady=5)

        def update_drivers(*args):
            drivers = env_preview.get_drivers_by_team(team_cb.get())
            driver_cb['values'] = drivers
            driver_cb.current(0)
        
        team_cb.bind("<<ComboboxSelected>>", update_drivers)
        update_drivers()

        tk.Label(frame, text="Grid (1-20):", fg="white", bg="#0a0d1a").grid(row=3, column=0, sticky="w", pady=5)
        grid_cb = ttk.Combobox(frame, values=["AUTO"] + [str(i) for i in range(1, 21)], width=30, state="readonly")
        grid_cb.current(0); grid_cb.grid(row=3, column=1, pady=5)

        def start():
            result.update({"team": team_cb.get(), "driver": driver_cb.get(), "race": race_cb.get(), "grid_pos": grid_cb.get()})
            root.destroy()

        tk.Button(root, text="CONFIRMAR CORRIDA", command=start, bg="#ff8000", fg="black", font=(FONT_NAME, 10, "bold")).pack(pady=20)
        root.mainloop()
        return result

    except Exception:
        pass

    # INTERFACE DE TERMINAL (Fallback principal)
    print("\n" + "="*60)
    print("F1 RACE SIMULATOR - MODO TERMINAL")
    print("="*60)
    
    # Escolher Circuito
    print("\nCIRCUITOS DISPONÍVEIS:")
    for i, race in enumerate(races, 1):
        print(f"  {i}. {race}")
    while True:
        try:
            race_idx = int(input("\nEscolha o circuito (número): ")) - 1
            if 0 <= race_idx < len(races):
                chosen_race = races[race_idx]
                break
            print("Opção inválida!")
        except ValueError:
            print("Digite um número válido!")
    
    # Escolher Equipe
    print("\nEQUIPES DISPONÍVEIS:")
    for i, team in enumerate(teams, 1):
        print(f"  {i}. {team}")
    while True:
        try:
            team_idx = int(input("\nEscolha a equipe (número): ")) - 1
            if 0 <= team_idx < len(teams):
                chosen_team = teams[team_idx]
                break
            print("Opção inválida!")
        except ValueError:
            print("Digite um número válido!")
    
    # Escolher Piloto
    drivers = env_preview.get_drivers_by_team(chosen_team)
    print(f"\nPILOTOS DA {chosen_team.upper()}:")
    for i, driver in enumerate(drivers, 1):
        print(f"  {i}. {driver}")
    while True:
        try:
            driver_idx = int(input("\nEscolha o piloto (número): ")) - 1
            if 0 <= driver_idx < len(drivers):
                chosen_driver = drivers[driver_idx]
                break
            print("Opção inválida!")
        except ValueError:
            print("Digite um número válido!")
    
    # Escolher Posição na Grid
    print("\nPOSIÇÃO NA GRID (1-20 ou 0 para AUTO):")
    while True:
        try:
            grid_input = int(input("Escolha a posição (número): "))
            if grid_input == 0 or (1 <= grid_input <= 20):
                chosen_grid = "AUTO" if grid_input == 0 else str(grid_input)
                break
            print("Opção inválida! Digite 0 (AUTO) ou 1-20")
        except ValueError:
            print("Digite um número válido!")
    
    print(f"\n✓ INICIANDO CORRIDA:")
    print(f"  Circuito: {chosen_race}")
    print(f"  Equipe: {chosen_team}")
    print(f"  Piloto: {chosen_driver}")
    print(f"  Grid: {chosen_grid}\n")
    
    return {"team": chosen_team, "driver": chosen_driver, "race": chosen_race, "grid_pos": chosen_grid}

def run_selected_race(agent_team, agent_driver, race_name, starting_position=None, interval=120):
    env = F1RaceEnv(
        data_path=DATA_DIR,
        team_perf_path=os.path.join(DATA_DIR, "team_performances.csv"),
        drivers_path=os.path.join(DATA_DIR, "drivers.csv"),
        race_name=race_name,
        agent_team=agent_team,
        agent_driver=agent_driver,
        starting_position=starting_position,
        render_mode=None,
    )

    obs, info = env.reset(seed=123)
    
    # Configuração da Figura
    fig = plt.figure(figsize=(16, 9), facecolor="#070b18")
    ax = fig.add_axes([0.05, 0.1, 0.65, 0.78])
    header_ax = fig.add_axes([0.05, 0.89, 0.9, 0.08])
    right_ax = fig.add_axes([0.72, 0.1, 0.25, 0.78])

    # Estética
    comp_colors = {"SOFT": "#ff2d2d", "MEDIUM": "#ffd400", "HARD": "#f3f3f3"}

    def draw_frame(frame_idx):
        if not hasattr(env, 'done') or not env.done:
            # Lógica simples de pit stop automática para o agente
            action = 0 
            if env.tyre_life < 30: action = 1 # Exemplo
            obs, reward, terminated, truncated, step_info = env.step(action)
            env.done = terminated or truncated
        else:
            step_info = {"lap": env.total_laps, "driver_history": env.driver_history, "live_gaps": env._get_live_gaps()}

        # Limpeza
        ax.clear(); header_ax.clear(); right_ax.clear()
        for a in [ax, header_ax, right_ax]: a.set_facecolor("#0a0d1a")
        header_ax.set_axis_off(); right_ax.set_axis_off()

        history = step_info["driver_history"]
        live_gaps = step_info.get("live_gaps", {})

        # Plotagem das linhas de corrida
        leaderboard = []
        for driver, d_hist in history.items():
            if not d_hist: continue
            laps = [p["lap"] for p in d_hist]
            pos = [p["position"] for p in d_hist]
            team = d_hist[-1]["team"]
            color = TEAM_COLORS.get(team, "#ffffff")
            is_agent = (driver == agent_driver)
            
            ax.plot(laps, pos, color=color, lw=3 if is_agent else 1.5, alpha=1.0 if is_agent else 0.7)
            
            # Marcador de Pit
            for p in d_hist:
                if p.get("pit"):
                    ax.scatter(p["lap"], p["position"], color=comp_colors.get(p["compound"], "white"), s=50, edgecolors="white", zorder=5)

            leaderboard.append({
                "pos": pos[-1], "name": driver[:12], "team": team, "color": color, 
                "gap": live_gaps.get(driver, {}).get("gap_to_leader", 0.0), "is_agent": is_agent
            })

        # Estilo do Eixo Principal
        ax.set_ylim(20.5, 0.5); ax.set_xlim(1, env.total_laps)
        ax.set_yticks(range(1, 21))
        ax.tick_params(colors="white", labelsize=9)
        ax.grid(True, alpha=0.1, color="white")
        ax.set_title(f"HISTÓRICO DE POSIÇÕES", color="white", loc="left", fontsize=10)

        # Header
        header_ax.text(0, 0.5, f"GP {race_name.upper()}", color="white", fontsize=18, fontweight="bold")
        header_ax.text(0.4, 0.5, f"VOLTA: {step_info['lap']}/{env.total_laps}", color="#ff8000", fontsize=16)

        # Leaderboard Lateral
        leaderboard.sort(key=lambda x: x["pos"])
        right_ax.set_ylim(20.5, 0.5); right_ax.set_xlim(0, 1)
        for i, d in enumerate(leaderboard):
            y = d["pos"]
            color = d["color"]
            weight = "bold" if d["is_agent"] else "normal"
            right_ax.text(0.05, y, f"{int(y):02d}", color=color, fontweight="bold", va="center")
            right_ax.text(0.20, y, d["name"].upper(), color="white", fontweight=weight, va="center", fontsize=9)
            gap_txt = "LEADER" if i == 0 else f"+{d['gap']:.3f}"
            right_ax.text(0.95, y, gap_txt, color="#ffd166", ha="right", va="center", fontsize=8)

    # CRÍTICO: Atribuir a uma variável para o Garbage Collector não deletar a animação
    ani = animation.FuncAnimation(fig, draw_frame, frames=env.total_laps, interval=interval, repeat=False)

    # Se estivermos em modo sem janela, salvamos automaticamente
    if matplotlib.get_backend() == 'Agg':
        save_path = os.path.join(OUT_DIR, "race_sim.gif")
        print(f"Salvando simulação em: {save_path}...")
        ani.save(save_path, writer='pillow')
    else:
        plt.show()

if __name__ == "__main__":
    setup = choose_race_setup()
    grid = None if setup["grid_pos"] == "AUTO" else int(setup["grid_pos"])
    
    run_selected_race(
        agent_team=setup["team"],
        agent_driver=setup["driver"],
        race_name=setup["race"],
        starting_position=grid
    )