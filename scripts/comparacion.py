"""
Gráfica comparativa Q-Learning vs DQN.

Requiere haber corrido antes `scripts/experimento.py` para ambos agentes
(usa results/<agente>/recompensas.csv y los agentes guardados en saves/).

Uso:
    uv run python scripts/comparacion.py
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experimento import CONFIG, ROOT, evaluar

COLORES = {"qlearning": "tab:blue", "dqn": "tab:orange"}
NOMBRES = {"qlearning": "Q-Learning", "dqn": "DQN"}


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 2]})

    evals = {}
    for nombre, cfg in CONFIG.items():
        rewards = np.loadtxt(ROOT / "results" / nombre / "recompensas.csv", skiprows=1)
        w = 100  # misma ventana para los dos, para que sean comparables
        ma = np.convolve(rewards, np.ones(w) / w, mode="valid")
        ax1.plot(np.arange(w, len(rewards) + 1), ma, color=COLORES[nombre], lw=2,
                 label=f"{NOMBRES[nombre]} ({len(rewards):,} episodios)".replace(",", "."))

        agent = cfg["cls"].load(cfg["save"])
        evals[nombre] = [r for r, _ in evaluar(agent, 100, seed=10_000)]

    ax1.axhline(-110, color="green", ls="--", lw=1.2, label="-110 (umbral de 'resuelto')")
    ax1.set_xscale("log")
    ax1.set(title="Entrenamiento: media móvil (100 episodios)", xlabel="Episodio (escala log)",
            ylabel="Recompensa total", ylim=(-205, -90))
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="upper left")

    datos = [evals[n] for n in CONFIG]
    bp = ax2.boxplot(datos, patch_artist=True, widths=0.5)
    ax2.set_xticks([1, 2], [NOMBRES[n] for n in CONFIG])
    for patch, n in zip(bp["boxes"], CONFIG):
        patch.set_facecolor(COLORES[n])
        patch.set_alpha(0.6)
    for i, n in enumerate(CONFIG, start=1):
        media = np.mean(evals[n])
        ax2.scatter([i], [media], color="black", zorder=5, marker="D")
        ax2.annotate(f"media {media:.1f}", (i, media), xytext=(12, 0), textcoords="offset points",
                     va="center")
    ax2.axhline(-110, color="green", ls="--", lw=1.2)
    ax2.set(title="Evaluación greedy (100 episodios)", ylabel="Recompensa total", ylim=(-205, -70))
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("MountainCar-v0 — Q-Learning vs DQN", fontsize=14)
    fig.tight_layout()
    out = ROOT / "results" / "comparacion.png"
    fig.savefig(out, dpi=150)
    print(f"Guardada: {out}")
    for n in CONFIG:
        print(f"{NOMBRES[n]}: media {np.mean(evals[n]):.2f} ± {np.std(evals[n]):.2f}")


if __name__ == "__main__":
    main()
