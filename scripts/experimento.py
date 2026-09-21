"""
Entrena un agente, lo evalúa y genera la evidencia de resultados.

Uso:
    uv run python scripts/experimento.py qlearning --episodes 20000
    uv run python scripts/experimento.py dqn --episodes 2500

Genera, dentro de results/<agente>/:
    recompensas.csv        recompensa de cada episodio de entrenamiento
    curva_entrenamiento.png recompensa por episodio + media móvil
    evaluacion.png         recompensas de la evaluación greedy
    resumen.json           hiperparámetros y métricas finales
y guarda el agente en saves/ (la misma ruta que usa el CLI), de modo que
después funcionan `mountaincar load <agente> --eval` y `mountaincar render <agente>`.
"""
import argparse
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mountain_car.agents import DQNAgent, QLearningAgent

ENV_ID = "MountainCar-v0"
ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "qlearning": {
        "cls": QLearningAgent,
        "save": ROOT / "saves" / "qlearning_mountaincar.pkl",
        "hparams": {},  # valores por defecto de QLearningAgent
        "window": 200,
    },
    "dqn": {
        "cls": DQNAgent,
        "save": ROOT / "saves" / "dqn_mountaincar.pt",
        "hparams": {},  # valores por defecto de DQNAgent
        "window": 100,
    },
}


def evaluar(agent, n: int, seed: int) -> list[tuple[float, bool]]:
    """Juega n episodios greedy (sin exploración)."""
    env = gym.make(ENV_ID)
    out = []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        total, done, terminated = 0.0, False, False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            done = terminated or truncated
        out.append((total, bool(terminated)))
    env.close()
    return out


def graficar(nombre: str, rewards: list[float], window: int, evals: list[float], out: Path) -> None:
    window = min(window, len(rewards))
    titulo = "Q-Learning" if nombre == "qlearning" else "DQN"
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    xs = np.arange(window, len(rewards) + 1)
    best = int(np.argmax(ma))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(1, len(rewards) + 1), rewards, color="tab:blue", alpha=0.25, lw=0.8,
            label="Recompensa por episodio")
    ax.plot(xs, ma, color="tab:orange", lw=2.2, label=f"Media móvil ({window} episodios)")
    ax.axhline(-110, color="green", ls="--", lw=1.2, label="-110 (umbral de 'resuelto')")
    ax.axhline(-200, color="gray", ls=":", lw=1, label="-200 (nunca llega a la bandera)")
    ax.scatter([xs[best]], [ma[best]], color="red", zorder=5)
    ax.annotate(f"Mejor media móvil: {ma[best]:.1f}\n(episodio {xs[best]})",
                (xs[best], ma[best]), xytext=(-170, 25), textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "red"}, color="red")
    ax.set(title=f"MountainCar-v0 — entrenamiento {titulo}", xlabel="Episodio",
           ylabel="Recompensa total", ylim=(-205, -80))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "curva_entrenamiento.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(1, len(evals) + 1), [-r for r in evals], color="tab:purple", alpha=0.8)
    mean = -float(np.mean(evals))
    ax.axhline(mean, color="black", ls="--", lw=1.2, label=f"Media: {mean:.1f} pasos (recompensa {-mean:.1f})")
    ax.axhline(110, color="green", ls="--", lw=1.2, label="110 pasos (umbral de 'resuelto')")
    ax.set(title=f"{titulo} — evaluación greedy ({len(evals)} episodios, sin exploración)",
           xlabel="Episodio de evaluación", ylabel="Pasos hasta la bandera (= −recompensa)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / "evaluacion.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("agent", choices=tuple(CONFIG))
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = CONFIG[args.agent]
    agent = cfg["cls"](ENV_ID, **cfg["hparams"])
    log_interval = 500 if args.agent == "qlearning" else 50

    t0 = time.time()
    rewards = agent.train(total_episodes=args.episodes, log_interval=log_interval)
    minutos = (time.time() - t0) / 60
    agent.save(cfg["save"])

    results = evaluar(agent, args.eval_episodes, seed=10_000)
    evals = [r for r, _ in results]
    llegadas = sum(ok for _, ok in results)

    out = ROOT / "results" / args.agent
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "recompensas.csv", rewards, fmt="%.0f", header="recompensa", comments="")
    graficar(args.agent, rewards, cfg["window"], evals, out)

    w = min(cfg["window"], len(rewards))
    ma = np.convolve(rewards, np.ones(w) / w, mode="valid")
    resumen = {
        "agente": args.agent,
        "episodios_entrenamiento": args.episodes,
        "semilla": args.seed,
        "minutos_entrenamiento": round(minutos, 1),
        "hiperparametros": {k: getattr(agent, k) for k in agent._HPARAMS},
        "mejor_media_movil_entrenamiento": round(float(ma.max()), 2),
        "media_ultimos_episodios_entrenamiento": round(float(np.mean(rewards[-w:])), 2),
        "evaluacion_episodios": args.eval_episodes,
        "evaluacion_media": round(float(np.mean(evals)), 2),
        "evaluacion_desviacion": round(float(np.std(evals)), 2),
        "evaluacion_mejor": float(max(evals)),
        "evaluacion_peor": float(min(evals)),
        "evaluacion_llegadas_bandera": f"{llegadas}/{args.eval_episodes}",
    }
    (out / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
