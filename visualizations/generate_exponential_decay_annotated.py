import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    # Distance normalized by the maximum expected distance M, so x in [0, 1].
    x = np.linspace(0.0, 1.0, 600)
    y = np.exp(-3.0 * x)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)

    ax.plot(x, np.exp(-1.0 * x), color="orange", lw=2.2, ls="--", label=r"$k=1$ (mierne)")
    ax.plot(x, y, color="deepskyblue", lw=2.6, label=r"$k=3$ (použité)")
    ax.plot(x, np.exp(-5.0 * x), color="forestgreen", lw=2.2, ls=":", label=r"$k=5$ (prísne)")

    # Reference helper levels and key points to explain score behavior.
    levels = [0.8, 0.5, 0.2, np.exp(-3.0)]
    for lvl in levels:
        ax.axhline(lvl, color="gray", lw=0.9, alpha=0.5, ls="--")

    key_x = [0.1, 0.3, 0.6, 1.0]
    key_y = [np.exp(-3.0 * t) for t in key_x]
    ax.scatter(key_x, key_y, color="black", edgecolors="white", s=28, zorder=5)

    ax.annotate(
        "Malá chyba vzdialenosti\n=> vysoké skóre",
        xy=(0.1, np.exp(-0.3)),
        xytext=(0.18, 0.9),
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=13,
    )
    ax.annotate(
        "Stredná oblasť\n=> skóre rýchlo klesá",
        xy=(0.3, np.exp(-0.9)),
        xytext=(0.42, 0.62),
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=13,
    )
    ax.annotate(
        "Veľká vzdialenosť\n=> silná penalizácia",
        xy=(0.6, np.exp(-1.8)),
        xytext=(0.7, 0.28),
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=13,
    )
    ax.annotate(
        r"Pri $d=M$: $s=e^{-3}\approx0.05$",
        xy=(1.0, np.exp(-3.0)),
        xytext=(0.62, 0.08),
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=13,
    )

    # ax.set_title("Exponenciálne mapovanie vzdialenosti na skóre", color="black", fontsize=21)
    ax.set_xlabel(r"Normalizovaná vzdialenosť $d/M$", color="black", fontsize=16)
    ax.set_ylabel(r"Skóre podobnosti $s(d)$", color="black", fontsize=16)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.08)

    ax.grid(True, color="#808080", alpha=0.4, ls="-", lw=0.7)
    ax.legend(facecolor="#F7F7F7", edgecolor="#777777", labelcolor="black", fontsize=15)

    # Use a light visual style with high contrast text.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.tick_params(colors="black", labelsize=14)

    out_path = "/home/emmika/School/diplomovka/dplmvk-txt/images/Plot-exponential-decay-k-comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
