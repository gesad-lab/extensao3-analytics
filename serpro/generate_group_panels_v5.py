"""Versao 5: paineis por grupo com foco em visibilidade.

Melhorias principais:
- Fontes e rotulos maiores
- Maior contraste visual
- Sintese analitica em palavras-chave

Saida:
- Um PNG por grupo em serpro/paineis_por_grupo_v5_apresentacao/
"""

from __future__ import annotations

import csv
import re
import textwrap
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("serpro/20260408-AnáliseDados - analysis.csv")
OUTPUT_DIR = Path("serpro/paineis_por_grupo_v5")

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


def wrap_lines(lines: list[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width) or [line])
    return wrapped


def clean_analysis_text(text: str) -> str:
    """
    Remove trechos de enunciado duplicados presentes em 'Análise'.
    1) Exclui tudo que inicia em '##' e vai até imediatamente antes de 'Início do semestre'.
    2) Mantém somente o trecho após a expressão-chave 'Análise:' (ou 'Analise:').
    """
    if not text:
        return ""
    cleaned = re.sub(r"##.*?(?=In[ií]cio do semestre)", "", text, flags=re.DOTALL)
    match = re.search(r"An[aá]lise\s*:\s*(.*)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        cleaned = match.group(1)
    cleaned = cleaned.strip(" \n:-")
    return cleaned if cleaned else text.strip()


def compact_question_map(rows: list[dict[str, str]], width: int = 130) -> str:
    labels = [f"Q{i + 1}: {short_question(r['Perguntas'])}" for i, r in enumerate(rows)]
    joined = "   ".join(labels)
    return "\n".join(textwrap.wrap(joined, width=width))


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def style_card(ax, title: str) -> None:
    ax.set_facecolor(COLORS["panel"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])
        spine.set_linewidth(1.2)
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", color=COLORS["title"], pad=10)


def keyword_trend(row: dict[str, str]) -> str:
    d_con = parse_percent(row[COLS["variacao"]["concorda"]])
    d_neu = parse_percent(row[COLS["variacao"]["neutro"]])
    d_dis = parse_percent(row[COLS["variacao"]["discorda"]])

    k = []
    k.append("C+" if d_con >= 0 else "C-")
    k.append("N+" if d_neu >= 0 else "N-")
    k.append("D+" if d_dis >= 0 else "D-")

    if d_con > 3 and d_neu < 0:
        k.append("Confiança↑")
    elif d_con < -3 and d_neu > 0:
        k.append("Indecisão↑")
    elif d_con > 0 and d_dis > 0 and d_neu < 0:
        k.append("Polarização")
    else:
        k.append("Ajuste moderado")

    amplitude = max(abs(d_con), abs(d_neu), abs(d_dis))
    if amplitude >= 10:
        k.append("Forte")
    elif amplitude >= 5:
        k.append("Média")
    else:
        k.append("Leve")

    return " | ".join(k)


def build_group_panel(group_name: str, rows: list[dict[str, str]], output_file: Path) -> None:
    n_questions = len(rows)

    # Escala maior para apresentacao.
    width = max(18.0, 8.0 + n_questions * 2.0)
    height = 14.0

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titleweight": "bold",
        }
    )

    fig = plt.figure(figsize=(width, height), constrained_layout=False)
    fig.patch.set_facecolor(COLORS["bg"])

    # Novo layout inspirado no rascunho: titulo/subtitulo, lista de questões, dois gráficos lado a lado, síntese em faixa inferior.
    grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.45, 2.0, 1.05],
        left=0.04,
        right=0.98,
        top=0.83,
        bottom=0.05,
        hspace=0.34,
    )

    ax_q = fig.add_subplot(grid[0])
    charts = grid[1].subgridspec(1, 2, width_ratios=[1.7, 1], wspace=0.18)
    ax_main = fig.add_subplot(charts[0])
    ax_var = fig.add_subplot(charts[1])
    ax_syn = fig.add_subplot(grid[2])

    style_card(ax_main, "Gráfico de Barras: Início vs Final")
    style_card(ax_var, "Gráfico de Variação Média (pp)")
    style_card(ax_syn, "Síntese Analítica")

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

    bar_w = 0.72
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
    ax_main.set_ylabel("Percentual (%)", fontsize=17)
    ax_main.set_xticks(x_positions)
    ax_main.set_xticklabels(labels, fontsize=14)
    ax_main.tick_params(axis="y", labelsize=14)
    ax_main.grid(axis="y", alpha=0.24)

    # Faixas discretas por pergunta para facilitar leitura em tela.
    for i in range(n_questions):
        left = 2 * i - 0.48
        right = 2 * i + 1.48
        ax_main.axvspan(left, right, color="#000000", alpha=0.025)

    # Rótulos internos com fonte maior.
    for x, d, n, c in zip(x_positions, dis_vals, neu_vals, con_vals):
        if d >= 12:
            ax_main.text(x, d / 2, f"{d:.1f}%", ha="center", va="center", fontsize=11, color="white")
        if n >= 12:
            ax_main.text(x, d + n / 2, f"{n:.1f}%", ha="center", va="center", fontsize=11, color="#1f2937")
        if c >= 12:
            ax_main.text(x, d + n + c / 2, f"{c:.1f}%", ha="center", va="center", fontsize=11, color="white")

    # Legenda maior, horizontal.
    ax_main.legend(
        loc="upper left",
        bbox_to_anchor=(0, 1.05),
        ncol=3,
        fontsize=13,
        frameon=True,
        framealpha=0.96,
        edgecolor=COLORS["border"],
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
    ax_var.set_yticklabels(names, fontsize=16)
    ax_var.tick_params(axis="x", labelsize=14)
    ax_var.grid(axis="x", alpha=0.25)

    lim = max(5.0, max(abs(v) for v in vals) + 2.2)
    ax_var.set_xlim(-lim, lim)

    for yi, v in zip(y, vals):
        ha = "left" if v >= 0 else "right"
        dx = 0.22 if v >= 0 else -0.22
        arrow = "▲" if v >= 0 else "▼"
        ax_var.text(v + dx, yi, f"{arrow} {v:+.2f} pp", va="center", ha=ha, fontsize=16, color="#1d2939")

    # Sintese analitica por palavras-chave.
    ax_syn.set_xticks([])
    ax_syn.set_yticks([])
    ax_syn.set_facecolor(COLORS["panel"])
    for spine in ax_syn.spines.values():
        spine.set_color(COLORS["border"])
        spine.set_linewidth(1.2)

    header_lines = wrap_lines(
        [f"Média do grupo: C {avg_con:+.2f} pp | N {avg_neu:+.2f} pp | D {avg_dis:+.2f} pp"],
        width=95,
    )

    # Síntese agora exibe a íntegra da coluna \"Análise\" do CSV para cada questão.
    lines: list[str] = []
    for idx, row in enumerate(rows):
        analysis = row.get("Análise") or row.get("Analise") or ""
        analysis = clean_analysis_text(analysis)
        if not analysis:
            continue
        q_lines = wrap_lines([f"Q{idx + 1}: {analysis}"], width=95)
        lines.extend(q_lines)
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()

    ax_syn.text(
        0.02,
        0.95,
        "\n".join(lines),
        transform=ax_syn.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        color="#1f2937",
    )

    fig.suptitle(
        "Evolução das Percepções sobre IA Generativa na Engenharia de Software",
        fontsize=31,
        fontweight="bold",
        color=COLORS["title"],
        y=0.965,
    )

    subtitle = (
        f"Grupo: {group_name} — Categorias: Concorda (azul), Neutro (cinza), Discorda (vermelho) | "
        "Variação em pontos percentuais (pp)"
    )
    subtitle_wrapped = "\n".join(textwrap.wrap(subtitle, width=110))

    fig.text(
        0.5,
        0.915,
        subtitle_wrapped,
        ha="center",
        va="center",
        fontsize=17,
        color=COLORS["sub"],
    )

    fig.subplots_adjust(top=0.875)

    # Faixa de perguntas, em lista linear.
    ax_q.set_xticks([])
    ax_q.set_yticks([])
    ax_q.set_facecolor("#f9fafb")
    ax_q.set_xlim(0, 1)
    ax_q.set_ylim(0, 1)

    # Área das questões mais alta e com fonte grande; caixa única ocupando toda a largura.
    line_y = 0.97
    gap = 0.30
    text_width = 85

    box_height = gap * n_questions + 0.08
    ax_q.add_patch(
        plt.Rectangle(
            (0.01, line_y - box_height + 0.03),
            0.98,
            box_height,
            facecolor="#f9fafb",
            edgecolor=COLORS["border"],
            linewidth=1.3,
        )
    )

    for i, row in enumerate(rows):
        text = f"Q{i + 1}: {row['Perguntas']}"
        wrapped = "\n".join(wrap_lines([text], width=text_width))

        ax_q.text(
            0.025,
            line_y - i * gap,
            wrapped,
            ha="left",
            va="top",
            fontsize=20,
            color=COLORS["title"],
            bbox=None,
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
        out = OUTPUT_DIR / f"painel_{slugify(group_name)}_v5_projecao.png"
        build_group_panel(group_name, group_rows, out)
        print(f"OK: {out}")

    print(f"\nConcluído: {len(groups)} painéis gerados em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
