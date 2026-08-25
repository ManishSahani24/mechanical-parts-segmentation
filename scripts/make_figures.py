"""Regenerate the README figures from the recorded 60-epoch training log.

The numbers in TRAIN_LOSS / VAL_LOSS / VAL_DICE are transcribed from the
submitted Kaggle run (2x Tesla T4, 60 epochs) and are also written out as
assets/training_history.csv so the charts are reproducible from data.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --- palette -------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8f8e88"
GRID = "#e6e5e1"
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
NEUTRAL = "#c3c2bb"

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# --- recorded run --------------------------------------------------------
TRAIN_LOSS = [
    2.6603, 1.6808, 1.1665, 0.7691, 0.5947, 0.5009, 0.4343, 0.4030, 0.3642, 0.3430,
    0.3156, 0.3053, 0.2950, 0.2843, 0.2832, 0.3940, 0.3868, 0.3610, 0.3503, 0.3191,
    0.3117, 0.3018, 0.2850, 0.2780, 0.2679, 0.2669, 0.2559, 0.2468, 0.2352, 0.2390,
    0.2373, 0.2279, 0.2221, 0.2186, 0.2112, 0.2064, 0.2017, 0.2034, 0.1959, 0.1988,
    0.1919, 0.1900, 0.1936, 0.1893, 0.1917, 0.2614, 0.2760, 0.2661, 0.2605, 0.2418,
    0.2461, 0.2355, 0.2366, 0.2364, 0.2275, 0.2201, 0.2189, 0.2242, 0.2178, 0.2141,
]
VAL_LOSS = [
    15.0860, 1.0971, 0.5739, 0.3230, 0.2060, 0.1543, 0.1310, 0.1178, 0.1118, 0.1073,
    0.0915, 0.0897, 0.0873, 0.0844, 0.0840, 0.1832, 0.1252, 0.1033, 0.1019, 0.0925,
    0.0860, 0.0816, 0.0809, 0.1044, 0.0776, 0.0719, 0.0664, 0.0639, 0.0659, 0.0643,
    0.0649, 0.0619, 0.0590, 0.0578, 0.0572, 0.0556, 0.0557, 0.0539, 0.0534, 0.0533,
    0.0528, 0.0523, 0.0520, 0.0520, 0.0519, 0.1346, 0.0748, 0.0716, 0.0668, 0.0644,
    0.0656, 0.0705, 0.0599, 0.0600, 0.0584, 0.0567, 0.0559, 0.0551, 0.0562, 0.0542,
]
VAL_DICE = [
    0.0250, 0.4183, 0.6638, 0.8071, 0.8819, 0.9118, 0.9264, 0.9330, 0.9369, 0.9369,
    0.9478, 0.9481, 0.9496, 0.9517, 0.9516, 0.8960, 0.9259, 0.9397, 0.9383, 0.9456,
    0.9499, 0.9517, 0.9534, 0.9379, 0.9541, 0.9577, 0.9602, 0.9623, 0.9612, 0.9616,
    0.9619, 0.9631, 0.9648, 0.9654, 0.9663, 0.9674, 0.9673, 0.9683, 0.9686, 0.9686,
    0.9690, 0.9692, 0.9694, 0.9694, 0.9695, 0.9250, 0.9548, 0.9580, 0.9599, 0.9618,
    0.9612, 0.9568, 0.9644, 0.9645, 0.9649, 0.9661, 0.9664, 0.9673, 0.9667, 0.9671,
]
RESTART_EPOCHS = [16, 46]  # first epoch after each cosine warm restart

# Per-class pixel share, sampled from 200 training masks
CLASS_SHARE = [
    ("background", 75.7),
    ("bolt", 6.2),
    ("spring", 6.1),
    ("ball_bearing", 4.2),
    ("washer", 3.0),
    ("o_ring", 2.6),
    ("hex_nut", 2.2),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def training_curves(path: Path) -> None:
    epochs = list(range(1, len(VAL_DICE) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)

    # --- losses ---
    ax1.plot(epochs, TRAIN_LOSS, color=SERIES_1, linewidth=2, label="Train (incl. aux heads)")
    ax1.plot(epochs, VAL_LOSS, color=SERIES_2, linewidth=2, label="Validation (main head)")
    ax1.set_ylim(0, 0.75)  # epoch-1 val loss of 15.1 is clipped; noted in the caption
    ax1.set_title("Dice + Focal loss", color=INK, fontsize=11, loc="left", pad=10)
    ax1.set_xlabel("Epoch", color=INK_2, fontsize=9)
    legend = ax1.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK_2)
    _style(ax1)

    # --- dice ---
    ax2.plot(epochs, VAL_DICE, color=SERIES_1, linewidth=2, zorder=3)
    for epoch in RESTART_EPOCHS:
        ax2.axvline(epoch, color=NEUTRAL, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax2.text(
            epoch + 0.8, 0.905, "LR restart", color=MUTED, fontsize=8,
            rotation=90, va="bottom",
        )

    best_epoch = VAL_DICE.index(max(VAL_DICE)) + 1
    best = max(VAL_DICE)
    ax2.scatter([best_epoch], [best], s=42, color=SERIES_1, zorder=4,
                edgecolor=SURFACE, linewidth=2)
    ax2.annotate(
        f"best {best:.4f} (ep {best_epoch})",
        xy=(best_epoch - 0.6, best - 0.002),
        xytext=(best_epoch - 5, best - 0.026),
        color=INK, fontsize=9, ha="right", va="center",
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9,
                        shrinkA=4, shrinkB=2),
    )
    ax2.set_ylim(0.88, 0.985)
    ax2.set_title("Validation Dice (6 foreground classes)", color=INK, fontsize=11,
                  loc="left", pad=10)
    ax2.set_xlabel("Epoch", color=INK_2, fontsize=9)
    _style(ax2)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def class_distribution(path: Path) -> None:
    names = [n for n, _ in CLASS_SHARE][::-1]
    values = [v for _, v in CLASS_SHARE][::-1]
    colors = [NEUTRAL if n == "background" else SERIES_1 for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 3.4), facecolor=SURFACE)
    bars = ax.barh(names, values, color=colors, height=0.62, zorder=2)
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.9, bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%", va="center", color=INK_2, fontsize=9,
        )

    ax.set_xlim(0, 85)
    ax.set_xlabel("Share of all pixels", color=INK_2, fontsize=9)
    ax.set_title(
        "Class imbalance: background is 76% of every image",
        color=INK, fontsize=11, loc="left", pad=10,
    )
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": range(1, len(VAL_DICE) + 1),
            "train_loss": TRAIN_LOSS,
            "val_loss": VAL_LOSS,
            "val_dice": VAL_DICE,
        }
    ).to_csv(ASSETS / "training_history.csv", index=False)
    print(f"Wrote {ASSETS / 'training_history.csv'}")

    training_curves(ASSETS / "training_curves.png")
    class_distribution(ASSETS / "class_distribution.png")


if __name__ == "__main__":
    main()
