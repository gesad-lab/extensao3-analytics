"""Versao 2: paineis por grupo com visual mais limpo e legivel.

Saida:
- Um PNG por grupo em serpro/paineis_por_grupo_v2/
"""

from __future__ import annotations

import csv
import textwrap
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("serpro/20260408-AnáliseDados - analysis.csv")
OUTPUT_DIR = Path("serpro/paineis_por_grupo_v2")

COLORS = {
    "concorda": "#1f77b4",
    "neutro": "#aeb7c2",
    "discorda": "#d62728",
    "ganho": "#1f77b4",
    "perda": "#d62728",
    "bg": "#edf2f7",
    "card": "#f8fafc",
    "border": "#d6dee8",
    "title": "#1a212c",
    "subtitle": "#3e4b5c",
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


def short_question(question: str, max_chars: int = 58) -> str:
    question = " ".join(question.split())
    if len(question) <= max_chars:
        return question
    return question[: max_chars - 3].rstrip() + "..."


def analysis_sentence(raw_analysis: str) -> str:
    marker = "Análise:"
    if marker in raw_analysis:
        section = raw_analysis.split(marker, maxsplit=1)[1].strip()
        sentence = section.split(".", maxsplit=1)[0].strip()
        if sentence:
            return sentence + "."
    compact = " ".join(raw_analysis.split())
    return compact[:150] + ("..." if len(compact) > 150 else "")


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def style_card(ax, title: str) -> None:
    ax.set_facecolor(COLORS["card"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])
        spine.set_linewidth(1.0)
    ax.set_title(
        title,
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=COLORS["title"],
        pad=8,
    )


def extract_series(rows: list[dict[str, str]], stage: str, bucket: str) -> list[float]:
    return [parse_percent(r[COLS[stage][bucket]]) for r in rows]


def build_group_panel(group_name: str, rows: list[dict[str, str]], output_file: Path) -> None:
    n_questions = len(rows)

    width = max(16.0, 6.2 + n_questions * 1.95)
    height = 9.7

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )

    fig = plt.figure(figsize=(width, height), constrained_layout=False)
    fig.patch.set_facecolor(COLORS["bg"])

    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[2.45, 1.35],
        height_ratios=[1.05, 1.05],
        left=0.04,
        right=0.985,
        top=0.88,
        bottom=0.08,
        wspace=0.12,
        hspace=0.17,
    )

    ax_main = fig.add_subplot(grid[:, 0])
    ax_var = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, 1])

    style_card(ax_main, "1) Percepções Gerais: Início vs Final")
    style_card(ax_var, "2) Resumo da Variação Semestral (pp)")
    style_card(ax_text, "3) Síntese Analítica")

    start_dis = extract_series(rows, "inicio", "discorda")
    start_neu = extract_series(rows, "inicio", "neutro")
    start_con = extract_series(rows, "inicio", "concorda")

    end_dis = extract_series(rows, "final", "discorda")
    end_neu = extract_series(rows, "final", "neutro")
    end_con = extract_series(rows, "final", "concorda")

    var_dis = extract_series(rows, "variacao", "discorda")
    var_neu = extract_series(rows, "variacao", "neutro")
    var_con = extract_series(rows, "variacao", "concorda")

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
    ax_main.set_ylabel("Percentual (%)", fontsize=11)
    ax_main.set_xticks(x_positions)
    ax_main.set_xticklabels(labels, fontsize=9)
    ax_main.grid(axis="y", alpha=0.28)

    # Cartela visual por pergunta
    for i in range(n_questions):
        left = 2 * i - 0.48
        right = 2 * i + 1.48
        ax_main.axvspan(left, right, color="#000000", alpha=0.03)

    ax_main.legend(
        loc="upper right",
        ncol=3,
        fontsize=9,
        frameon=True,
        framealpha=0.92,
        edgecolor=COLORS["border"],
    )

    for x, d, n, c in zip(x_positions, dis_vals, neu_vals, con_vals):
        if d >= 10:
            ax_main.text(x, d / 2, f"{d:.1f}%", ha="center", va="center", fontsize=8, color="white")
        if n >= 10:
            ax_main.text(x, d + n / 2, f"{n:.1f}%", ha="center", va="center", fontsize=8, color="#222")
        if c >= 10:
            ax_main.text(x, d + n + c / 2, f"{c:.1f}%", ha="center", va="center", fontsize=8, color="white")

    # Variação média
    avg_dis = float(np.mean(var_dis))
    avg_neu = float(np.mean(var_neu))
    avg_con = float(np.mean(var_con))

    names = ["Concorda", "Neutro", "Discorda"]
    vals = [avg_con, avg_neu, avg_dis]
    y = np.arange(3)
    colors = [COLORS["ganho"] if v >= 0 else COLORS["perda"] for v in vals]

    ax_var.barh(y, vals, color=colors, alpha=0.9)
    ax_var.axvline(0, color="#4f5d6e", lw=1)
    ax_var.set_yticks(y)
    ax_var.set_yticklabels(names, fontsize=11)
    ax_var.grid(axis="x", alpha=0.25)
    lim = max(5.0, max(abs(v) for v in vals) + 2.0)
    ax_var.set_xlim(-lim, lim)

    for yi, v in zip(y, vals):
        ha = "left" if v >= 0 else "right"
        dx = 0.18 if v >= 0 else -0.18
        signal = "▲" if v >= 0 else "▼"
        ax_var.text(v + dx, yi, f"{signal} {v:+.2f} pp", ha=ha, va="center", fontsize=10, color="#1f2937")

    # Bloco textual harmonizado
    ax_text.set_xticks([])
    ax_text.set_yticks([])

    delta_con = np.array(var_con)
    top_idx = np.argsort(np.abs(delta_con))[::-1]
    top_k = 3 if n_questions >= 5 else min(4, n_questions)
    selected = sorted(top_idx[:top_k])

    insights = []
    for idx in selected:
        sentence = analysis_sentence(rows[idx].get("Análise", ""))
        insights.append(f"Q{idx + 1}: {sentence}")

    if n_questions > top_k:
        insights.append(f"Outras {n_questions - top_k} questões mantiveram padrão semelhante.")

    wrapped_lines = []
    for line in insights:
        wrapped_lines.extend(textwrap.wrap(line, width=54) or [line])

    questions_block = [f"Q{i + 1}: {short_question(r['Perguntas'], 50)}" for i, r in enumerate(rows)]
    question_map = "\n".join(questions_block)
    question_map = "Mapa de Questões\n" + question_map

    summary = (
        f"Média do grupo\n"
        f"Concorda {avg_con:+.2f} pp | Neutro {avg_neu:+.2f} pp | Discorda {avg_dis:+.2f} pp"
    )

    ax_text.text(
        0.03,
        0.95,
        summary,
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#1f2937",
        fontweight="bold",
    )

    ax_text.text(
        0.03,
        0.78,
        "\n".join(wrapped_lines),
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#2a3340",
    )

    ax_text.text(
        0.03,
        0.03,
        "\n".join(textwrap.wrap(question_map, width=60)),
        transform=ax_text.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#465466",
    )

    fig.suptitle(
        "Evolução das Percepções sobre Utilidade e Produtividade com IA Generativa\n"
        f"Grupo: {group_name}",
        fontsize=18,
        fontweight="bold",
        color=COLORS["title"],
        y=0.972,
    )

    fig.text(
        0.5,
        0.905,
        "Categorias: Concorda (azul), Neutro (cinza), Discorda (vermelho) | Variação positiva/negativa em pp",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["subtitle"],
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
        file_name = f"painel_{slugify(group_name)}_v2.png"
        out = OUTPUT_DIR / file_name
        build_group_panel(group_name, group_rows, out)
        print(f"OK: {out}")

    print(f"\nConcluído: {len(groups)} painéis gerados em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
