from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mountain_car.agents.qlearning import QLearningAgent

agent = QLearningAgent("MountainCar-v0", n_bins=24, lr=0.2, epsilon_decay=0.995)
rewards = agent.train(total_episodes=5000, log_interval=50)

save_path = ROOT / "plots"
save_path.mkdir(exist_ok=True)

plt.figure(figsize=(12, 6))
plt.plot(rewards, color="tab:blue", linewidth=1.5)
plt.axhline(-200, color="gray", linestyle="--", linewidth=1, alpha=0.7)
plt.axhline(-110, color="green", linestyle="--", linewidth=1, alpha=0.7)
plt.title("MountainCar - Q-learning reward curve")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_path / "qlearning_rewards.png", dpi=200)
print(f"Saved plot: {save_path / 'qlearning_rewards.png'}")

window = 200
moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
plt.figure(figsize=(12, 6))
plt.plot(range(window - 1, len(rewards)), moving_avg, color="tab:green", linewidth=2)
plt.axhline(-110, color="red", linestyle="--", linewidth=1, alpha=0.7)
plt.title("MountainCar - moving average reward (window=200)")
plt.xlabel("Episode")
plt.ylabel("Moving average reward")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_path / "qlearning_moving_avg.png", dpi=200)
print(f"Saved plot: {save_path / 'qlearning_moving_avg.png'}")
