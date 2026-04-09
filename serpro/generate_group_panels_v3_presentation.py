"""Versao 3 (apresentacao): paineis por grupo com foco em visibilidade.

Melhorias principais:
- Fontes e rotulos maiores
- Maior contraste visual
- Sintese analitica em palavras-chave

Saida:
- Um PNG por grupo em serpro/paineis_por_grupo_v3_apresentacao/
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("serpro/20260408-AnáliseDados - analysis.csv")
OUTPUT_DIR = Path("serpro/paineis_por_grupo_v3_apresentacao")

COLORS = {
    "concorda": "#1f77b4",
    "neutro": "#8f98a3",
    "discorda": "#d62728",
    "gain": "#1f77b4",
    "loss": "#d62728",
    "bg": "#ffffff",
    "panel": "#f7f9fc",
    "border": "#d5dde7",
    "title": "#101828",
    "sub": "#344054",
}

COLS = {
    "inicio": {
        "discorda": "[INÍCIO] % Discorda ou Discorda Totalmente (<3)",
        "neutro": "[INÍCIO] % Neutro (=3)",
        "concorda": "[INÍCIO] % Concorda ou Concorda Totalmente (>3)",
    },
    "final": {
        "discorda": "[FINAL] % Discorda ou Discorda Totalmente (<3)",
        "neutro": "[FINAL] % Neutro (=3)",
        "concorda": "[FINAL] % Concorda ou Concorda Totalmente (>3)",
    },
    "variacao": {
        "discorda": "[VARIAÇÃO] % Discorda ou Discorda Totalmente (<3)",
        "neutro": "[VARIAÇÃO] % Neutro (=3)",
        "concorda": "[VARIAÇÃO] % Concorda ou Concorda Totalmente (>3)",
    },
}


def parse_percent(value: str) -> float:
    return float(value.strip().replace("%", "").replace(",", "."))


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in ascii_text)
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-")


def short_question(question: str, max_chars: int = 48) -> str:
    q = " ".join(question.split())
    if len(q) <= max_chars:
        return q
    return q[: max_chars - 3].rstrip() + "..."


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def style_card(ax, title: str) -> None:
    ax.set_facecolor(COLORS["panel"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])
        spine.set_linewidth(1.2)
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color=COLORS["title"], pad=10)


def keyword_trend(row: dict[str, str]) -> str:
    d_con = parse_percent(row[COLS["variacao"]["concorda"]])
    d_neu = parse_percent(row[COLS["variacao"]["neutro"]])
    d_dis = parse_percent(row[COLS["variacao"]["discorda"]])

    k = []
    k.append("Concorda+" if d_con >= 0 else "Concorda-")
    k.append("Neutro+" if d_neu >= 0 else "Neutro-")
    k.append("Discorda+" if d_dis >= 0 else "Discorda-")

    if d_con > 3 and d_neu < 0:
        k.append("Maior confiança")
    elif d_con < -3 and d_neu > 0:
        k.append("Maior indecisão")
    elif d_con > 0 and d_dis > 0 and d_neu < 0:
        k.append("Polarização")
    else:
        k.append("Ajuste moderado")

    amplitude = max(abs(d_con), abs(d_neu), abs(d_dis))
    if amplitude >= 10:
        k.append("Mudança forte")
    elif amplitude >= 5:
        k.append("Mudança média")
    else:
        k.append("Mudança leve")

    return " | ".join(k)


def build_group_panel(group_name: str, rows: list[dict[str, str]], output_file: Path) -> None:
    n_questions = len(rows)

    # Escala maior para apresentacao.
    width = max(18.0, 8.0 + n_questions * 2.15)
    height = 11.2

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titleweight": "bold",
        }
    )

    fig = plt.figure(figsize=(width, height), constrained_layout=False)
    fig.patch.set_facecolor(COLORS["bg"])

    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[2.5, 1.35],
        height_ratios=[1.05, 1.15],
        left=0.035,
        right=0.985,
        top=0.865,
        bottom=0.08,
        wspace=0.12,
        hspace=0.18,
    )

    ax_main = fig.add_subplot(grid[:, 0])
    ax_var = fig.add_subplot(grid[0, 1])
    ax_key = fig.add_subplot(grid[1, 1])

    style_card(ax_main, "1) Percepções Gerais: Início vs Final")
    style_card(ax_var, "2) Variação Média do Grupo (pp)")
    style_card(ax_key, "3) Síntese por Palavras-chave")

    start_dis = [parse_percent(r[COLS["inicio"]["discorda"]]) for r in rows]
    start_neu = [parse_percent(r[COLS["inicio"]["neutro"]]) for r in rows]
    start_con = [parse_percent(r[COLS["inicio"]["concorda"]]) for r in rows]

    end_dis = [parse_percent(r[COLS["final"]["discorda"]]) for r in rows]
    end_neu = [parse_percent(r[COLS["final"]["neutro"]]) for r in rows]
    end_con = [parse_percent(r[COLS["final"]["concorda"]]) for r in rows]

    x_positions = []
    labels = []
    for i in range(1, n_questions + 1):
        x_positions.extend([2 * (i - 1), 2 * (i - 1) + 1])
        labels.extend([f"Q{i} I", f"Q{i} F"])

    dis_vals = np.ravel(np.column_stack([start_dis, end_dis]))
    neu_vals = np.ravel(np.column_stack([start_neu, end_neu]))
    con_vals = np.ravel(np.column_stack([start_con, end_con]))

    bar_w = 0.76
    ax_main.bar(x_positions, dis_vals, width=bar_w, color=COLORS["discorda"], label="Discorda")
    ax_main.bar(x_positions, neu_vals, width=bar_w, bottom=dis_vals, color=COLORS["neutro"], label="Neutro")
    ax_main.bar(
        x_positions,
        con_vals,
        width=bar_w,
        bottom=dis_vals + neu_vals,
        color=COLORS["concorda"],
        label="Concorda",
    )

    ax_main.set_ylim(0, 100)
    ax_main.set_ylabel("Percentual (%)", fontsize=15)
    ax_main.set_xticks(x_positions)
    ax_main.set_xticklabels(labels, fontsize=12)
    ax_main.tick_params(axis="y", labelsize=12)
    ax_main.grid(axis="y", alpha=0.24)

    # Faixas discretas por pergunta para facilitar leitura em tela.
    for i in range(n_questions):
        left = 2 * i - 0.48
        right = 2 * i + 1.48
        ax_main.axvspan(left, right, color="#000000", alpha=0.025)

    # Rótulos internos com fonte maior.
    for x, d, n, c in zip(x_positions, dis_vals, neu_vals, con_vals):
        if d >= 12:
            ax_main.text(x, d / 2, f"{d:.1f}%", ha="center", va="center", fontsize=10, color="white")
        if n >= 12:
            ax_main.text(x, d + n / 2, f"{n:.1f}%", ha="center", va="center", fontsize=10, color="#1f2937")
        if c >= 12:
            ax_main.text(x, d + n + c / 2, f"{c:.1f}%", ha="center", va="center", fontsize=10, color="white")

    # Legenda maior, horizontal.
    ax_main.legend(
        loc="upper right",
        ncol=3,
        fontsize=12,
        frameon=True,
        framealpha=0.95,
        edgecolor=COLORS["border"],
    )

    # Linha de apoio com texto de questoes resumidas.
    q_labels = [f"Q{i + 1}: {short_question(r['Perguntas'])}" for i, r in enumerate(rows)]
    ax_main.text(
        0.005,
        -0.115,
        "   ".join(q_labels),
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=COLORS["sub"],
    )

    # Painel de variacao media
    avg_dis = float(np.mean([parse_percent(r[COLS["variacao"]["discorda"]]) for r in rows]))
    avg_neu = float(np.mean([parse_percent(r[COLS["variacao"]["neutro"]]) for r in rows]))
    avg_con = float(np.mean([parse_percent(r[COLS["variacao"]["concorda"]]) for r in rows]))

    names = ["Concorda", "Neutro", "Discorda"]
    vals = [avg_con, avg_neu, avg_dis]
    y = np.arange(len(names))
    colors = [COLORS["gain"] if v >= 0 else COLORS["loss"] for v in vals]

    ax_var.barh(y, vals, color=colors, alpha=0.9)
    ax_var.axvline(0, color="#475467", lw=1.2)
    ax_var.set_yticks(y)
    ax_var.set_yticklabels(names, fontsize=14)
    ax_var.tick_params(axis="x", labelsize=12)
    ax_var.grid(axis="x", alpha=0.25)

    lim = max(5.0, max(abs(v) for v in vals) + 2.2)
    ax_var.set_xlim(-lim, lim)

    for yi, v in zip(y, vals):
        ha = "left" if v >= 0 else "right"
        dx = 0.22 if v >= 0 else -0.22
        arrow = "▲" if v >= 0 else "▼"
        ax_var.text(v + dx, yi, f"{arrow} {v:+.2f} pp", va="center", ha=ha, fontsize=14, color="#1d2939")

    # Sintese analitica por palavras-chave.
    ax_key.set_xticks([])
    ax_key.set_yticks([])

    # Seleciona ate 5 perguntas com maior mudanca em concorda.
    idx_sorted = np.argsort([abs(parse_percent(r[COLS["variacao"]["concorda"]])) for r in rows])[::-1]
    top_n = min(5, n_questions)
    selected = sorted(idx_sorted[:top_n])

    lines = [
        f"Média do grupo: Concorda {avg_con:+.2f} pp | Neutro {avg_neu:+.2f} pp | Discorda {avg_dis:+.2f} pp",
        "",
    ]

    for idx in selected:
        lines.append(f"Q{idx + 1}: {keyword_trend(rows[idx])}")

    if n_questions > top_n:
        lines.append(f"+{n_questions - top_n} questão(ões): padrão semelhante")

    ax_key.text(
        0.03,
        0.95,
        "\n".join(lines),
        transform=ax_key.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color="#1f2937",
    )

    fig.suptitle(
        "Evolução das Percepções sobre IA Generativa na Engenharia de Software\n"
        f"Grupo: {group_name}",
        fontsize=25,
        fontweight="bold",
        color=COLORS["title"],
        y=0.972,
    )

    fig.text(
        0.5,
        0.905,
        "Categorias: Concorda (azul), Neutro (cinza), Discorda (vermelho) | Variação em pontos percentuais (pp)",
        ha="center",
        va="center",
        fontsize=14,
        color=COLORS["sub"],
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=220)
    plt.close(fig)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}")

    rows = load_rows()
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["Grupo"], []).append(row)

    for group_name, group_rows in groups.items():
        out = OUTPUT_DIR / f"painel_{slugify(group_name)}_v3_apresentacao.png"
        build_group_panel(group_name, group_rows, out)
        print(f"OK: {out}")

    print(f"\nConcluído: {len(groups)} painéis gerados em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
