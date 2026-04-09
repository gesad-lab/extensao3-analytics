from pathlib import Path
import re
import subprocess
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

pd.set_option("display.max_columns", 200)
plt.rcParams["figure.figsize"] = (10, 6)

REPO_DIR = Path.cwd().parent
BASE = REPO_DIR / "base_files"
OUT = REPO_DIR / "EDA" / "eda_outputs_multi_cohort_v2"
OUT.mkdir(parents=True, exist_ok=True)
PAT = re.compile(r"^responses(\d+)\.csv$", re.I)

ID_COLS = ["Email Address", "Nome completo"]
FREQ = {
    "Nunca": 0,
    "Raramente": 1,
    "Algumas vezes por mês": 2,
    "Semanalmente": 3,
    "Diariamente": 4,
}
LIKERT = {
    "Discordo totalmente": 1,
    "Discordo": 2,
    "Neutro": 3,
    "Concordo": 4,
    "Concordo totalmente": 5,
}

GENERAL = "Com que frequência você utiliza ferramentas de IA generativa em seu dia a dia de maneira geral?"
SE = "Com que frequência você utiliza ferramentas de IA Generativa especificamente em projetos de engenharia de software?"
TOOLS = "Quais ferramentas de IA generativa você utilizou nos últimos 12 meses em projetos de engenharia de software? (selecione todas que se aplicam)"
LIKERT_COLS = [
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [Ferramentas de IA generativa em Engenharia de Software são úteis.]",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [IA generativa aumenta minha produtividade na engenharia de software.]",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [IA generativa melhora a qualidade do software produzido.]",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [O uso de IA generativa altera o fluxo de desenvolvimento de sistemas.]",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [Existe metodologia mais adequada às ferramentas de IA generativa.]",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [O uso de ferramentas de IA generativa exige novas competências que não fazem parte da formação tradicional em Engenharia de Software.]",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [Novos papéis profissionais surgirão devido à IA generativa (por exemplo, AI Prompt Engineer, AI Auditor).]",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [Sinto-me confiante em revisar e integrar código gerado por IA generativa.]",
]


def git_sync():
    if not (REPO_DIR / ".git").exists():
        return "skipped"
    try:
        subprocess.run(
            ["git", "fetch", "--all", "--prune"],
            cwd=REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=REPO_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return "ok"
    except Exception as exc:
        return f"failed: {exc}"


def dedup(df):
    for dup in [c for c in df.columns if str(c).endswith(".1")]:
        base = dup[:-2]
        if base in df.columns and df[base].notna().sum() == 0 and df[dup].notna().sum() > 0:
            df[base] = df[dup]
            df = df.drop(columns=[dup])
    return df


def load_all(base_dir=BASE):
    if not base_dir.exists():
        raise FileNotFoundError("Expected base_files/<cohort>/responses<stage>.csv")

    frames = []
    rows = []
    for cohort_dir in sorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        files = sorted(
            [p for p in cohort_dir.iterdir() if p.is_file() and PAT.match(p.name)],
            key=lambda p: (int(PAT.match(p.name).group(1)), p.name),
        )
        for csv in files:
            stage = int(PAT.match(csv.name).group(1))
            df = dedup(pd.read_csv(csv))
            df["cohort"] = cohort_dir.name
            df["stage"] = stage
            frames.append(df)
            rows.append(
                {
                    "cohort": cohort_dir.name,
                    "stage": stage,
                    "file_name": csv.name,
                    "relative_path": csv.as_posix(),
                    "n_rows": len(df),
                    "n_columns": df.shape[1],
                }
            )

    if not frames:
        raise FileNotFoundError(
            "No files matching responses<stage>.csv were found under base_files/<cohort>"
        )

    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True, sort=False)


def split_counts(series):
    items = []
    for cell in series.dropna():
        items.extend([x.strip() for x in str(cell).split(",") if x.strip()])
    return pd.Series(items, dtype="object").value_counts() if items else pd.Series(dtype=int)


def barh(series, title, out_name, normalize=True, top_n=None):
    s = series.dropna()
    if s.empty:
        return
    counts = s.value_counts()
    if top_n:
        counts = counts.head(top_n)
    values = counts / counts.sum() * 100 if normalize else counts
    plt.figure(figsize=(10, max(4, len(counts) * 0.45)))
    plt.barh(counts.index.astype(str), values.values)
    for i, x in enumerate(values.values):
        plt.text(x + 0.2, i, f"{x:.1f}%" if normalize else str(int(x)), va="center")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()


print("Git sync:", git_sync())
manifest, raw = load_all()
print(f"Loaded {len(manifest)} source file(s)")
print(f"Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
display(manifest)

raw = raw.drop(columns=[c for c in ID_COLS if c in raw.columns]).copy()
raw["general_num"] = raw[GENERAL].map(FREQ) if GENERAL in raw.columns else np.nan
raw["se_num"] = raw[SE].map(FREQ) if SE in raw.columns else np.nan
num_cols = []
for i, c in enumerate([c for c in LIKERT_COLS if c in raw.columns]):
    n = f"likert_{i}"
    raw[n] = raw[c].map(LIKERT)
    num_cols.append(n)
raw["perception_index"] = raw[num_cols].mean(axis=1) if num_cols else np.nan

schema = pd.DataFrame(
    {
        "column": raw.columns,
        "dtype": raw.dtypes.astype(str).values,
        "missing_pct": (raw.isna().mean().values * 100).round(2),
    }
)
display(schema)
schema.to_csv(OUT / "schema_audit.csv", index=False)

cohort_stage = raw.groupby(["cohort", "stage"]).size().reset_index(name="n_responses")
display(cohort_stage)
cohort_stage.to_csv(OUT / "cohort_stage_summary.csv", index=False)
raw.to_csv(OUT / "cleaned_anonymized_multi_cohort_data.csv", index=False)

if GENERAL in raw.columns:
    barh(raw[GENERAL], "General GenAI usage frequency", "general_genai_usage.png")
    usage_by_group = pd.crosstab(
        raw["cohort"].astype(str) + " | stage " + raw["stage"].astype(str),
        raw[GENERAL],
        normalize="index",
    ) * 100
    display(usage_by_group.round(1))
    usage_by_group.to_csv(OUT / "general_genai_usage_by_group.csv")

if SE in raw.columns:
    barh(raw[SE], "GenAI use in software engineering projects", "se_genai_usage.png")

if TOOLS in raw.columns:
    top_tools = split_counts(raw[TOOLS]).head(12)
    if not top_tools.empty:
        plt.figure(figsize=(10, max(4, len(top_tools) * 0.45)))
        plt.barh(top_tools.index.astype(str), top_tools.values)
        plt.title("Top GenAI tools")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUT / "top_genai_tools.png", dpi=160, bbox_inches="tight")
        plt.show()
        plt.close()
        top_tools.rename_axis("tool").reset_index(name="count").to_csv(
            OUT / "top_genai_tools.csv", index=False
        )

if num_cols:
    item_means = pd.DataFrame(
        {
            "item": [c for c in LIKERT_COLS if c in raw.columns],
            "mean": [raw[c].mean() for c in num_cols],
        }
    ).sort_values("mean", ascending=False)
    display(item_means)
    item_means.to_csv(OUT / "likert_item_means.csv", index=False)

if "perception_index" in raw.columns:
    perception_by_group = (
        raw.groupby(["cohort", "stage"])["perception_index"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    display(perception_by_group)
    perception_by_group.to_csv(OUT / "perception_by_group.csv", index=False)

stats = []
xy = raw[["general_num", "perception_index"]].dropna()
if len(xy) >= 5:
    rho, p_value = spearmanr(xy["general_num"], xy["perception_index"])
    stats.append(
        {
            "test": "spearman",
            "x": "general_num",
            "y": "perception_index",
            "stat": rho,
            "p_value": p_value,
            "n": len(xy),
        }
    )

xy = raw[["se_num", "perception_index"]].dropna()
if len(xy) >= 5:
    rho, p_value = spearmanr(xy["se_num"], xy["perception_index"])
    stats.append(
        {
            "test": "spearman",
            "x": "se_num",
            "y": "perception_index",
            "stat": rho,
            "p_value": p_value,
            "n": len(xy),
        }
    )

cohort_groups = [
    g["perception_index"].dropna().values
    for _, g in raw.groupby("cohort")
    if g["perception_index"].notna().sum() >= 3
]
if len(cohort_groups) == 2:
    stat, p_value = mannwhitneyu(cohort_groups[0], cohort_groups[1], alternative="two-sided")
    stats.append(
        {
            "test": "mannwhitney",
            "x": "cohort",
            "y": "perception_index",
            "stat": stat,
            "p_value": p_value,
        }
    )

stage_groups = [
    g["perception_index"].dropna().values
    for _, g in raw.groupby("stage")
    if g["perception_index"].notna().sum() >= 3
]
if len(stage_groups) >= 2:
    stat, p_value = kruskal(*stage_groups)
    stats.append(
        {
            "test": "kruskal",
            "x": "stage",
            "y": "perception_index",
            "stat": stat,
            "p_value": p_value,
        }
    )

stats = pd.DataFrame(stats)
display(stats)
stats.to_csv(OUT / "stats_summary.csv", index=False)

print("Outputs written to", OUT)
