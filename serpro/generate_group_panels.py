"""Gera paineis por grupo a partir do CSV de analise da pesquisa.

Saida:
- Um PNG por grupo em serpro/paineis_por_grupo/
"""

from __future__ import annotations

import csv
import textwrap
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("serpro/20260408-AnáliseDados - analysis.csv")
OUTPUT_DIR = Path("serpro/paineis_por_grupo")

# Cores base do painel
COLORS = {
    "concorda": "#1f77b4",  # Azul
    "neutro": "#b0b0b0",    # Cinza
    "discorda": "#d62728",  # Vermelho
    "positivo": "#1f77b4",  # Ganho
    "negativo": "#d62728",  # Perda
    "fundo": "#f5f7f9",
    "borda": "#d3dbe3",
    "titulo": "#1b1f24",
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
    """Converte percentual textual em float."""
    return float(value.strip().replace("%", "").replace(",", "."))


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in ascii_text)
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-")


def short_question(question: str, max_chars: int = 54) -> str:
    q = " ".join(question.split())
    if len(q) <= max_chars:
        return q
    return q[: max_chars - 3].rstrip() + "..."


def analysis_sentence(raw_analysis: str) -> str:
    """Extrai frase curta apos 'Análise:' para painel mais limpo."""
    marker = "Análise:"
    if marker in raw_analysis:
        section = raw_analysis.split(marker, maxsplit=1)[1].strip()
        sentence = section.split(".", maxsplit=1)[0].strip()
        if sentence:
            return sentence + "."
    compact = " ".join(raw_analysis.split())
    return compact[:140] + ("..." if len(compact) > 140 else "")


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def style_card(ax, title: str) -> None:
    ax.set_facecolor(COLORS["fundo"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["borda"])
        spine.set_linewidth(1.0)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=COLORS["titulo"], pad=8)


def build_group_panel(group_name: str, rows: list[dict[str, str]], output_file: Path) -> None:
    n_questions = len(rows)

    # Mais perguntas -> painel mais largo automaticamente.
    width = max(15.0, 3.4 + n_questions * 2.2)
    height = 9.2

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    fig.patch.set_facecolor("#eef2f6")

    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[2.35, 1.35],
        height_ratios=[1.25, 1.0],
    )

    ax_main = fig.add_subplot(grid[:, 0])
    ax_var = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, 1])

    style_card(ax_main, "Percepções Gerais: Início vs Final")
    style_card(ax_var, "Resumo da Variação Semestral (pp)")
    style_card(ax_text, "Análise Sintética")

    # -------------------------
    # 1) Barras empilhadas
    # -------------------------
    x_positions = []
    labels = []

    start_dis = []
    start_neu = []
    start_con = []

    end_dis = []
    end_neu = []
    end_con = []

    for i, row in enumerate(rows, start=1):
        x_positions.extend([2 * (i - 1), 2 * (i - 1) + 1])
        labels.extend([f"Q{i} I", f"Q{i} F"])

        start_dis.append(parse_percent(row[COLS["inicio"]["discorda"]]))
        start_neu.append(parse_percent(row[COLS["inicio"]["neutro"]]))
        start_con.append(parse_percent(row[COLS["inicio"]["concorda"]]))

        end_dis.append(parse_percent(row[COLS["final"]["discorda"]]))
        end_neu.append(parse_percent(row[COLS["final"]["neutro"]]))
        end_con.append(parse_percent(row[COLS["final"]["concorda"]]))

    dis_vals = np.ravel(np.column_stack([start_dis, end_dis]))
    neu_vals = np.ravel(np.column_stack([start_neu, end_neu]))
    con_vals = np.ravel(np.column_stack([start_con, end_con]))

    bar_w = 0.78
    ax_main.bar(x_positions, dis_vals, color=COLORS["discorda"], width=bar_w, label="Discorda")
    ax_main.bar(x_positions, neu_vals, bottom=dis_vals, color=COLORS["neutro"], width=bar_w, label="Neutro")
    ax_main.bar(
        x_positions,
        con_vals,
        bottom=dis_vals + neu_vals,
        color=COLORS["concorda"],
        width=bar_w,
        label="Concorda",
    )

    ax_main.set_ylim(0, 100)
    ax_main.set_ylabel("Percentual (%)", fontsize=10)
    ax_main.set_xticks(x_positions)
    ax_main.set_xticklabels(labels, rotation=0, fontsize=9)
    ax_main.grid(axis="y", alpha=0.28)
    ax_main.legend(loc="upper right", frameon=True, fontsize=9)

    # Separadores visuais entre perguntas
    for i in range(1, n_questions):
        ax_main.axvline(2 * i - 0.5, color="#c9d2dd", lw=0.8, ls="--", alpha=0.8)

    # Marcadores Q1, Q2, ... abaixo das barras
    for i, row in enumerate(rows, start=1):
        center = 2 * (i - 1) + 0.5
        ax_main.text(
            center,
            -9.5,
            f"Q{i}: {short_question(row['Perguntas'], 44)}",
            ha="center",
            va="top",
            fontsize=8,
            color="#2f3742",
            clip_on=False,
        )

    # Rotulos internos para segmentos maiores
    for x, d, n, c in zip(x_positions, dis_vals, neu_vals, con_vals):
        if d >= 8:
            ax_main.text(x, d / 2, f"{d:.1f}%", ha="center", va="center", fontsize=8, color="white")
        if n >= 8:
            ax_main.text(x, d + n / 2, f"{n:.1f}%", ha="center", va="center", fontsize=8, color="#222")
        if c >= 8:
            ax_main.text(x, d + n + c / 2, f"{c:.1f}%", ha="center", va="center", fontsize=8, color="white")

    # -------------------------
    # 2) Variação média do grupo
    # -------------------------
    var_dis = np.mean([parse_percent(r[COLS["variacao"]["discorda"]]) for r in rows])
    var_neu = np.mean([parse_percent(r[COLS["variacao"]["neutro"]]) for r in rows])
    var_con = np.mean([parse_percent(r[COLS["variacao"]["concorda"]]) for r in rows])

    var_names = ["Concorda", "Neutro", "Discorda"]
    var_vals = [var_con, var_neu, var_dis]

    y = np.arange(len(var_names))
    colors = [COLORS["positivo"] if v >= 0 else COLORS["negativo"] for v in var_vals]

    ax_var.barh(y, var_vals, color=colors, alpha=0.92)
    ax_var.axvline(0, color="#4c5663", lw=1.0)
    ax_var.set_yticks(y)
    ax_var.set_yticklabels(var_names, fontsize=10)
    lim = max(5.0, max(abs(v) for v in var_vals) + 2.0)
    ax_var.set_xlim(-lim, lim)
    ax_var.grid(axis="x", alpha=0.25)

    for yi, v in zip(y, var_vals):
        side = 0.16 if v >= 0 else -0.16
        ha = "left" if v >= 0 else "right"
        ax_var.text(v + side, yi, f"{v:+.2f} pp", va="center", ha=ha, fontsize=10, color="#1b1f24")

    # -------------------------
    # 3) Bloco de analise
    # -------------------------
    ax_text.set_xticks([])
    ax_text.set_yticks([])

    insights = []
    for i, row in enumerate(rows, start=1):
        sentence = analysis_sentence(row.get("Análise", ""))
        insights.append(f"Q{i}: {sentence}")

    # Mantem o bloco harmonico: no maximo 6 linhas principais
    if len(insights) > 4:
        shown = insights[:4]
        shown.append(f"... +{len(insights) - 4} questões com tendência semelhante.")
    else:
        shown = insights

    wrapped = []
    for line in shown:
        wrapped.extend(textwrap.wrap(line, width=52) or [line])

    summary_title = (
        f"Média do grupo: Concorda {var_con:+.2f} pp | "
        f"Neutro {var_neu:+.2f} pp | Discorda {var_dis:+.2f} pp"
    )

    full_text = [summary_title, ""] + wrapped
    ax_text.text(
        0.02,
        0.95,
        "\n".join(full_text),
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#222831",
    )

    # Titulo global
    fig.suptitle(
        "Evolução das Percepções sobre IA Generativa na Engenharia de Software\n"
        f"Grupo: {group_name}",
        fontsize=16,
        fontweight="bold",
        color=COLORS["titulo"],
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}")

    rows = load_rows()
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["Grupo"], []).append(row)

    for group_name, group_rows in groups.items():
        file_name = f"painel_{slugify(group_name)}.png"
        output_file = OUTPUT_DIR / file_name
        build_group_panel(group_name, group_rows, output_file)
        print(f"OK: {output_file}")

    print(f"\nConcluído: {len(groups)} painéis gerados em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
