from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt


RESULTS = [
    ("Random Agent", -5000),
    ("Early PPO", -3000),
    ("Improved PPO", -347),
    ("Reward-Shaped PPO", -135),
    ("PPO + NLP System", -15),
]


def main():
    labels = [name for name, _ in RESULTS]
    rewards = [reward for _, reward in RESULTS]

    output = Path("plots/reward_curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(labels, rewards, marker="o", linewidth=2.5, color="#2563eb")
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_title("OpenRescue AI Reward Improvement")
    ax.set_ylabel("Episode reward")
    ax.set_xlabel("System iteration")
    ax.grid(True, alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
