"""\n# Exploratory Data Analysis: Generative AI Adoption and Perception in Software Engineering

This notebook delivers a **deep, reproducible EDA** of the survey dataset in `responses1.csv`.

## Objective
Understand **adoption** and **perception** of generative AI in software engineering, with emphasis on:

- respondent profile and experience context
- overall and software-engineering-specific AI usage
- tools and activities associated with generative AI adoption
- perceived usefulness, productivity, quality, process change, and skill implications
- perceived impact on software engineering roles
- exploratory subgroup comparisons and light statistical testing

## Important notes
- Personally identifying columns are removed from the analytical dataset.
- Two duplicated fields with suffix `.1` are treated as import artifacts. The original versions are fully empty; the `.1` versions contain the actual responses.
- Open-ended text responses are **kept in the cleaned dataset** for completeness but **excluded from qualitative text analysis**, per project requirements.
- Statistical tests in this notebook are **exploratory**, not confirmatory, due to the small sample size.\n"""

# Core imports
from pathlib import Path
import textwrap
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, Markdown

from scipy.stats import (
    spearmanr,
    mannwhitneyu,
    kruskal,
    chi2_contingency
)

# Reproducibility / display config
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 200)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

DATA_PATH = Path("responses1.csv")
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# Load the raw dataset
raw = pd.read_csv(DATA_PATH)
print(f"Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
display(raw.head(3))


"""\n## 1. Initial schema audit

This section checks:
- dataset size
- column names
- duplicated fields introduced by import/export
- obvious missingness issues\n"""

schema = pd.DataFrame({
    "column": raw.columns,
    "dtype": raw.dtypes.astype(str).values,
    "non_null": raw.notna().sum().values,
    "missing": raw.isna().sum().values,
    "missing_pct": (raw.isna().mean().values * 100).round(2)
})
display(schema)


# Identify import-style duplicated columns, such as 'question' and 'question.1'
duplicate_like_cols = [c for c in raw.columns if c.endswith(".1")]
duplicate_audit = []

for dup_col in duplicate_like_cols:
    base_col = dup_col[:-2]
    base_exists = base_col in raw.columns
    duplicate_audit.append({
        "base_column": base_col,
        "duplicate_column": dup_col,
        "base_exists": base_exists,
        "base_non_null": int(raw[base_col].notna().sum()) if base_exists else np.nan,
        "duplicate_non_null": int(raw[dup_col].notna().sum()),
        "base_all_missing": bool(raw[base_col].isna().all()) if base_exists else np.nan
    })

duplicate_audit = pd.DataFrame(duplicate_audit)
display(duplicate_audit)

if not duplicate_audit.empty:
    print("Interpretation:")
    for _, row in duplicate_audit.iterrows():
        print(
            f"- Base column all missing? {row['base_all_missing']} | "
            f"Using populated duplicate: {row['duplicate_column']}"
        )


"""\n## 2. Data cleaning and anonymization

Cleaning decisions implemented below:
1. Remove direct identifiers (`Email Address`, `Nome completo`).
2. Parse `Timestamp`.
3. Trim whitespace in string cells.
4. Replace fully-empty base columns with populated `.1` counterparts.
5. Build an English alias map for easier downstream analysis.\n"""

df = raw.copy()

# Trim whitespace from string cells
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({"nan": np.nan, "": np.nan})

# Resolve duplicate import artifact columns
for dup_col in [c for c in df.columns if c.endswith(".1")]:
    base_col = dup_col[:-2]
    if base_col in df.columns and df[base_col].isna().all() and df[dup_col].notna().any():
        df[base_col] = df[dup_col]
        df = df.drop(columns=[dup_col])

# Parse timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# Remove direct identifiers
identifier_cols = [c for c in ["Email Address", "Nome completo"] if c in df.columns]
df_anon = df.drop(columns=identifier_cols).copy()

print("Removed identifier columns:", identifier_cols)
print(f"Anonymized shape: {df_anon.shape[0]} rows x {df_anon.shape[1]} columns")


# Column alias map for more readable analysis code
alias_map = {
    "Timestamp": "timestamp",
    "Em qual semestre do curso de Engenharia de Software você está?": "semester",
    "Você já possui outra graduação?": "prior_degree",
    "Qual é a sua faixa etária?": "age_band",
    "Qual é o seu gênero?": "gender",
    "Você concluiu o Ensino Médio em escola pública?": "public_high_school",
    "Quantos anos de experiência em desenvolvimento de software você possui?": "software_experience",
    "Você está empregado atualmente na área de software?": "employed_in_software",
    "Quantos projetos de engenharia de software você participou ao longo de sua carreira?": "project_count",
    "Qual papel você mais se identifica em projetos de software?": "primary_role",
    "Qual o número de membros da maior equipe de projeto em que você já atuou?": "largest_team_size",
    "Com que frequência você utiliza ferramentas de IA generativa em seu dia a dia de maneira geral?": "genai_general_frequency",
    "Com que frequência você utiliza ferramentas de IA Generativa especificamente em projetos de engenharia de software?": "genai_se_frequency",
    "Para quais atividades você costuma utilizar ferramentas de IA generativa? (marque todas as que se aplicam)": "genai_activities",
    "Quais ferramentas de IA generativa você utilizou nos últimos 12 meses em projetos de engenharia de software? (selecione todas que se aplicam)": "genai_tools_last_12m",
    "Qual metodologia de desenvolvimento de software você mais utilizou em seus projetos?": "methodology",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [Ferramentas de IA generativa em Engenharia de Software são úteis.]": "likert_useful",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [IA generativa aumenta minha produtividade na engenharia de software.]": "likert_productivity",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [IA generativa melhora a qualidade do software produzido.]": "likert_quality",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [O uso de IA generativa altera o fluxo de desenvolvimento de sistemas.]": "likert_flow_change",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [É possível utilizar qualquer metodologia de desenvolvimento de software com IA generativa.]": "likert_any_methodology",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [Existe metodologia mais adequada às ferramentas de IA generativa.]": "likert_methodology_fit",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [O uso de ferramentas de IA generativa exige novas competências que não fazem parte da formação tradicional em Engenharia de Software.]": "likert_new_competencies",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [Novos papéis profissionais surgirão devido à IA generativa (por exemplo, AI Prompt Engineer, AI Auditor).]": "likert_new_roles",
    "Avalie seu nível de concordância com as afirmações abaixo relacionadas ao uso de ferramentas de IA generativa em engenharia de software. [Sinto-me confiante em revisar e integrar código gerado por IA generativa.]": "likert_confidence_ai_code",
    "A função de backend será extinta ou severamente afetada pelo uso de ferramentas de IA generativa na engenharia de software. Indique seu grau de concordância de 1 (Discordo totalmente) a 5 (Concordo totalmente).": "role_backend_impact",
    "A função de frontend será extinta ou severamente afetada pelo uso de ferramentas de IA generativa na engenharia de software. Indique seu grau de concordância de 1 (Discordo totalmente) a 5 (Concordo totalmente).": "role_frontend_impact",
    "A função de QA (Garantia de Qualidade) será extinta ou severamente afetada pelo uso de ferramentas de IA generativa na engenharia de software. Indique seu grau de concordância de 1 (Discordo totalmente) a 5 (Concordo totalmente).": "role_qa_impact",
    "A função de gerente de projeto será extinta ou severamente afetada pelo uso de ferramentas de IA generativa na engenharia de software. Indique seu grau de concordância de 1 (Discordo totalmente) a 5 (Concordo totalmente).": "role_pmgr_impact",
    "A função de Product Manager será extinta ou severamente afetada pelo uso de ferramentas de IA generativa na engenharia de software. Indique seu grau de concordância de 1 (Discordo totalmente) a 5 (Concordo totalmente).": "role_product_manager_impact",
    "A função de Scrum Master será extinta ou severamente afetada pelo uso de ferramentas de IA generativa na engenharia de software. Indique seu grau de concordância de 1 (Discordo totalmente) a 5 (Concordo totalmente).": "role_scrum_master_impact",
    "Qual foi o maior benefício que você obteve (ou imagina) ao usar ferramentas de IA generativa em projetos de Engenharia de Software?": "open_biggest_benefit",
    "Em 5 anos, como você enxerga o impacto das ferramentas de IA generativa em sua carreira de engenharia de software?": "open_five_year_impact",
    "Antes de iniciar o projeto, quão confiante você está de que sua equipe vai entregar um software funcional e de qualidade dentro do prazo?": "project_confidence_pre_start",
    "Quais aspectos do projeto você acredita que serão mais desafiadores? (marque todos que se aplicam)": "project_challenges",
    "Acredita que as ferramentas de apoio e automação adotadas pela equipe irão agilizar significativamente o desenvolvimento?": "automation_speed_up",
    "Você prevê que será necessário revisar ou ajustar significativamente as saídas geradas por ferramentas automatizadas ao longo do projeto?": "automation_revision_needed",
    "Quão motivado você está para aprender novas técnicas e tecnologias durante a implementação do projeto?": "learning_motivation",
    "Na sua opinião, o aprendizado obtido durante este projeto será:": "learning_expected_level",
    "Você acredita que as ferramentas de apoio e automação adotadas pela equipe irão facilitar a colaboração entre os membros da sua equipe?": "automation_collaboration_help",
    "Para você, qual deve ser o equilíbrio ideal entre a autonomia de cada integrante e a dependência de ferramentas de suporte no desenvolvimento do projeto?": "autonomy_vs_tool_dependence",
    "Descreva brevemente como você se sente em relação ao projeto que irá desenvolver nesta disciplina.": "open_project_feeling",
    "Quais expectativas você tem sobre sua própria evolução profissional ao participar deste projeto?": "open_professional_growth",
    "Você já desenvolveu ou participou de projetos de desenvolvimento de software que usa IA Generativa para realizar alguma(s) funcionalidade(s) ": "prior_project_with_genai_feature",
    "Você leu o Termo de Consentimento Livre e Esclarecido acima e concorda com a coleta e o uso dos artefatos que serão gerados por você durante a disciplina para fins de pesquisa?": "consent"
}

analysis_df = df_anon.rename(columns=alias_map).copy()
missing_aliases = [c for c in analysis_df.columns if c not in alias_map.values()]
print("Columns without alias replacement:")
print(missing_aliases if missing_aliases else "None")
display(analysis_df.head(2))


# Ordinal maps and helper variables
frequency_map = {
    "Nunca": 0,
    "Raramente": 1,
    "Algumas vezes por mês": 2,
    "Semanalmente": 3,
    "Diariamente": 4
}

likert_map = {
    "Discordo totalmente": 1,
    "Discordo": 2,
    "Neutro": 3,
    "Concordo": 4,
    "Concordo totalmente": 5
}

binary_map = {"Não": 0, "Sim": 1}

software_experience_order = [
    "Nenhuma", "Menos de 1 ano", "1-3 anos", "3-5 anos", "Mais de 5 anos"
]
project_count_order = ["Nenhum", "1-2 projetos", "3-5 projetos", "Mais de 5 projetos"]
team_size_order = ["1-5", "6-10", "11-20", "21-50", "Mais de 50"]
age_order = ["17-20 anos", "21-25 anos", "26-30 anos", "31-40 anos", "Mais de 40 anos"]
semester_order = ["3º semestre", "4º semestre", "5º semestre", "6º semestre", "7º semestre", "8º semestre", "9º semestre"]
methodology_order = [
    "Ágil (Scrum, Kanban)", "Modelo híbrido", "Cascata", "Nenhuma metodologia aplicada", "Não sei / Não conheço"
]
learning_expected_order = [
    "Básico (vou consolidar conhecimentos já adquiridos)",
    "Moderado (aprenderei algumas novas técnicas)",
    "Significativo (aprenderei várias técnicas novas)",
    "Transformador (mudará minha forma de desenvolver software)"
]

analysis_df["genai_general_frequency_num"] = analysis_df["genai_general_frequency"].map(frequency_map)
analysis_df["genai_se_frequency_num"] = analysis_df["genai_se_frequency"].map(frequency_map)
analysis_df["prior_project_with_genai_feature_num"] = analysis_df["prior_project_with_genai_feature"].map(binary_map)
analysis_df["employed_in_software_num"] = analysis_df["employed_in_software"].map(binary_map)

likert_cols = [
    "likert_useful",
    "likert_productivity",
    "likert_quality",
    "likert_flow_change",
    "likert_any_methodology",
    "likert_methodology_fit",
    "likert_new_competencies",
    "likert_new_roles",
    "likert_confidence_ai_code",
]
for col in likert_cols:
    analysis_df[col + "_num"] = analysis_df[col].map(likert_map)

role_impact_cols = [
    "role_backend_impact",
    "role_frontend_impact",
    "role_qa_impact",
    "role_pmgr_impact",
    "role_product_manager_impact",
    "role_scrum_master_impact"
]

analysis_df["perception_index"] = analysis_df[[c + "_num" for c in likert_cols]].mean(axis=1)
analysis_df["role_disruption_index"] = analysis_df[role_impact_cols].mean(axis=1)


def split_multiselect(series, sep=","):
    tokens = []
    for cell in series.dropna():
        parts = [item.strip() for item in str(cell).split(sep)]
        tokens.extend([p for p in parts if p])
    return pd.Series(tokens)

def multiselect_counts(series, sep=","):
    s = split_multiselect(series, sep=sep)
    if s.empty:
        return pd.Series(dtype=int)
    return s.value_counts()

def count_selected(series, sep=","):
    return series.fillna("").apply(
        lambda x: len([item.strip() for item in str(x).split(sep) if item.strip()])
    )

analysis_df["n_genai_tools"] = count_selected(analysis_df["genai_tools_last_12m"])
analysis_df["n_genai_activities"] = count_selected(analysis_df["genai_activities"])
analysis_df["n_project_challenges"] = count_selected(analysis_df["project_challenges"])

analysis_df["high_se_usage"] = analysis_df["genai_se_frequency_num"] >= 3
analysis_df["daily_or_weekly_general_usage"] = analysis_df["genai_general_frequency_num"] >= 3


"""\n## 3. Data quality review

Focus:
- missingness
- row duplication
- variable type balance
- integrity of the cleaned analytical dataset\n"""

print("Duplicate rows in anonymized dataset:", analysis_df.duplicated().sum())
print("Rows with missing timestamp:", analysis_df["timestamp"].isna().sum())

missing_report = (
    analysis_df.isna()
    .mean()
    .sort_values(ascending=False)
    .rename("missing_pct")
    .mul(100)
    .round(2)
    .reset_index()
    .rename(columns={"index": "column"})
)
display(missing_report)


# Plot missingness percentages
missing_nonzero = missing_report[missing_report["missing_pct"] > 0].copy()

plt.figure(figsize=(10, max(4, len(missing_nonzero) * 0.35)))
if missing_nonzero.empty:
    plt.text(0.5, 0.5, "No missing values after cleaning", ha="center", va="center", fontsize=13)
    plt.axis("off")
else:
    plt.barh(missing_nonzero["column"], missing_nonzero["missing_pct"])
    plt.xlabel("Missing values (%)")
    plt.title("Missingness after cleaning")
    plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


"""\n### Interpretation
For this dataset, missingness is low in the structured questions after duplicate-field repair and anonymization.  
The main data quality issue is not missingness, but **schema irregularity** caused by the duplicated open-ended fields.\n"""

"""\n## 4. Respondent profile

This section characterizes the sample before focusing on AI-specific questions.\n"""

def plot_categorical_distribution(series, title, order=None, normalize=True, top_n=None):
    s = series.dropna()
    if order is not None:
        counts = s.value_counts().reindex(order, fill_value=0)
    else:
        counts = s.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    values = counts / counts.sum() * 100 if normalize else counts
    xlabel = "Percentage of respondents" if normalize else "Count"
    plt.figure(figsize=(10, max(4, len(counts) * 0.45)))
    plt.barh(counts.index.astype(str), values.values)
    for i, v in enumerate(values.values):
        suffix = "%" if normalize else ""
        plt.text(v + (0.3 if normalize else 0.05), i, f"{v:.1f}{suffix}" if normalize else f"{int(v)}", va="center")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


profile_columns = [
    ("semester", semester_order, "Semester distribution"),
    ("age_band", age_order, "Age band distribution"),
    ("gender", None, "Gender distribution"),
    ("public_high_school", ["Sim", "Não"], "Public high school background"),
    ("software_experience", software_experience_order, "Software development experience"),
    ("employed_in_software", ["Sim", "Não"], "Currently employed in software"),
    ("project_count", project_count_order, "Software engineering project count"),
    ("largest_team_size", team_size_order, "Largest project team size"),
    ("primary_role", None, "Primary role identity"),
    ("methodology", methodology_order, "Most-used development methodology"),
]

for col, order, title in profile_columns:
    plot_categorical_distribution(analysis_df[col], title=title, order=order, normalize=True)


profile_summary = pd.DataFrame({
    "metric": [
        "Sample size",
        "Median semester (approx.)",
        "Share with software employment",
        "Share with prior project using GenAI functionality",
        "Share using GenAI weekly/daily in general",
        "Share using GenAI weekly/daily in software engineering"
    ],
    "value": [
        len(analysis_df),
        analysis_df["semester"].str.extract(r"(\d+)").astype(float).median().iloc[0],
        round(analysis_df["employed_in_software_num"].mean() * 100, 1),
        round(analysis_df["prior_project_with_genai_feature_num"].mean() * 100, 1),
        round(analysis_df["daily_or_weekly_general_usage"].mean() * 100, 1),
        round(analysis_df["high_se_usage"].mean() * 100, 1),
    ]
})
display(profile_summary)


"""\n## 5. Adoption of generative AI

This section examines:
- overall usage frequency
- software-engineering-specific usage frequency
- activities supported by GenAI
- tools used in the last 12 months
- intensity of tool/activity diversification\n"""

plot_categorical_distribution(
    analysis_df["genai_general_frequency"],
    title="How often respondents use GenAI in general",
    order=["Nunca", "Raramente", "Algumas vezes por mês", "Semanalmente", "Diariamente"],
    normalize=True
)

plot_categorical_distribution(
    analysis_df["genai_se_frequency"],
    title="How often respondents use GenAI in software engineering projects",
    order=["Nunca", "Raramente", "Algumas vezes por mês", "Semanalmente", "Diariamente"],
    normalize=True
)


# Crosstab: general use vs software-engineering use
ct = pd.crosstab(
    analysis_df["genai_general_frequency"],
    analysis_df["genai_se_frequency"],
    normalize="index"
).reindex(
    ["Nunca", "Raramente", "Algumas vezes por mês", "Semanalmente", "Diariamente"]
).fillna(0)

display(ct.style.format("{:.2%}"))


# Multi-select: activities
activity_counts = multiselect_counts(analysis_df["genai_activities"])
display(activity_counts.to_frame("count"))

plt.figure(figsize=(10, max(4, len(activity_counts) * 0.45)))
plt.barh(activity_counts.index, activity_counts.values)
for i, v in enumerate(activity_counts.values):
    plt.text(v + 0.1, i, str(int(v)), va="center")
plt.title("Activities for which respondents use generative AI")
plt.xlabel("Number of respondents selecting the activity")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Multi-select: tools
tool_counts = multiselect_counts(analysis_df["genai_tools_last_12m"])
display(tool_counts.to_frame("count"))

plt.figure(figsize=(10, max(4, len(tool_counts) * 0.45)))
plt.barh(tool_counts.index, tool_counts.values)
for i, v in enumerate(tool_counts.values):
    plt.text(v + 0.1, i, str(int(v)), va="center")
plt.title("Generative AI tools used in software engineering projects (last 12 months)")
plt.xlabel("Number of respondents selecting the tool")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Number of tools and activities per respondent
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(analysis_df["n_genai_tools"], bins=np.arange(-0.5, analysis_df["n_genai_tools"].max() + 1.5, 1), rwidth=0.9)
axes[0].set_title("Distribution of number of GenAI tools used")
axes[0].set_xlabel("Number of tools")
axes[0].set_ylabel("Respondent count")

axes[1].hist(analysis_df["n_genai_activities"], bins=np.arange(-0.5, analysis_df["n_genai_activities"].max() + 1.5, 1), rwidth=0.9)
axes[1].set_title("Distribution of number of GenAI-supported activities")
axes[1].set_xlabel("Number of activities")
axes[1].set_ylabel("Respondent count")

plt.tight_layout()
plt.show()

display(
    analysis_df[["n_genai_tools", "n_genai_activities"]]
    .describe()
    .T
)


# Adoption by subgroup
def ordered_mean_by_group(df, group_col, value_col, order=None):
    out = df.groupby(group_col)[value_col].mean().sort_values(ascending=False)
    if order is not None:
        out = out.reindex(order)
    return out

subgroup_specs = [
    ("software_experience", software_experience_order, "Mean software-engineering GenAI usage by experience"),
    ("semester", semester_order, "Mean software-engineering GenAI usage by semester"),
    ("employed_in_software", ["Não", "Sim"], "Mean software-engineering GenAI usage by employment status"),
    ("prior_project_with_genai_feature", ["Não", "Sim"], "Mean software-engineering GenAI usage by prior GenAI-project experience"),
]

for group_col, order, title in subgroup_specs:
    means = ordered_mean_by_group(analysis_df, group_col, "genai_se_frequency_num", order=order)
    plt.figure(figsize=(10, max(4, len(means) * 0.45)))
    plt.barh(means.index.astype(str), means.values)
    for i, v in enumerate(means.values):
        plt.text(v + 0.03, i, f"{v:.2f}", va="center")
    plt.title(title)
    plt.xlabel("Mean usage score (0=Never, 4=Daily)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


"""\n## 6. Perception of generative AI in software engineering

This section converts the agreement statements into ordinal scores:
- 1 = strongly disagree
- 2 = disagree
- 3 = neutral
- 4 = agree
- 5 = strongly agree\n"""

likert_labels = {
    "likert_useful": "GenAI tools are useful",
    "likert_productivity": "GenAI increases my productivity",
    "likert_quality": "GenAI improves software quality",
    "likert_flow_change": "GenAI changes development flow",
    "likert_any_methodology": "Any methodology can work with GenAI",
    "likert_methodology_fit": "Some methodologies fit GenAI better",
    "likert_new_competencies": "GenAI requires new competencies",
    "likert_new_roles": "New professional roles will emerge",
    "likert_confidence_ai_code": "I feel confident reviewing/integrating AI-generated code"
}

likert_order = [
    "Discordo totalmente",
    "Discordo",
    "Neutro",
    "Concordo",
    "Concordo totalmente"
]

likert_dist = []
for col, label in likert_labels.items():
    row = analysis_df[col].value_counts(normalize=True).reindex(likert_order, fill_value=0)
    row.name = label
    likert_dist.append(row)

likert_dist = pd.DataFrame(likert_dist)
display(likert_dist.style.format("{:.1%}"))


# Horizontal stacked Likert chart
plot_df = likert_dist.copy()

left = np.zeros(len(plot_df))
plt.figure(figsize=(12, 7))
for response in likert_order:
    vals = plot_df[response].values
    plt.barh(plot_df.index, vals, left=left, label=response)
    left += vals

plt.title("Distribution of agreement across perception statements")
plt.xlabel("Proportion of respondents")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# Mean item scores
mean_item_scores = (
    analysis_df[[c + "_num" for c in likert_cols]]
    .mean()
    .rename(index={c + "_num": label for c, label in likert_labels.items()})
    .sort_values()
)

display(mean_item_scores.to_frame("mean_score"))

plt.figure(figsize=(10, 6))
plt.barh(mean_item_scores.index, mean_item_scores.values)
for i, v in enumerate(mean_item_scores.values):
    plt.text(v + 0.03, i, f"{v:.2f}", va="center")
plt.xlim(0, 5.4)
plt.xlabel("Mean agreement score (1-5)")
plt.title("Average agreement with GenAI perception statements")
plt.tight_layout()
plt.show()


# Perception index distribution
plt.figure(figsize=(9, 5))
plt.hist(analysis_df["perception_index"], bins=10, rwidth=0.9)
plt.title("Distribution of perception index")
plt.xlabel("Perception index (mean of 9 Likert items)")
plt.ylabel("Respondent count")
plt.tight_layout()
plt.show()

display(analysis_df["perception_index"].describe().to_frame("perception_index"))


"""\n### Interpretation
The perception index summarizes overall orientation toward GenAI in software engineering.  
It should be read as a **composite exploratory score**, not as a validated psychometric scale.\n"""

"""\n## 7. Perceived impact on software engineering roles

These items were already numeric on a 1–5 agreement scale and can be compared directly.\n"""

role_labels = {
    "role_backend_impact": "Backend",
    "role_frontend_impact": "Frontend",
    "role_qa_impact": "QA",
    "role_pmgr_impact": "Project manager",
    "role_product_manager_impact": "Product manager",
    "role_scrum_master_impact": "Scrum Master",
}

role_means = analysis_df[list(role_labels)].mean().rename(index=role_labels).sort_values()
display(role_means.to_frame("mean_impact_score"))

plt.figure(figsize=(10, 6))
plt.barh(role_means.index, role_means.values)
for i, v in enumerate(role_means.values):
    plt.text(v + 0.03, i, f"{v:.2f}", va="center")
plt.xlim(0, 5.4)
plt.xlabel("Mean perceived disruption/extinction impact (1-5)")
plt.title("Perceived impact of GenAI on software engineering roles")
plt.tight_layout()
plt.show()


# Distribution by role item
role_item_summary = analysis_df[list(role_labels)].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
role_item_summary.index = role_item_summary.index.map(role_labels)
display(role_item_summary)


"""\n## 8. Relationship analysis and exploratory statistical testing

Because the sample is modest (`n = 46`), the analysis uses non-parametric or contingency-based methods.

Tests included:
- **Spearman correlation** for ordinal/continuous-like association
- **Mann–Whitney U** for two-group comparisons
- **Kruskal–Wallis** for >2 group comparisons
- **Chi-square** and **Cramér's V** for contingency relationships

These results are exploratory and should be interpreted with caution.\n"""

def cramers_v(table):
    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    if n == 0:
        return np.nan
    return np.sqrt((chi2 / n) / max(1, min(k - 1, r - 1)))

def mannwhitney_compare(df, value_col, group_col, group_order):
    g1 = df.loc[df[group_col] == group_order[0], value_col].dropna()
    g2 = df.loc[df[group_col] == group_order[1], value_col].dropna()
    stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
    return {
        "group_col": group_col,
        "value_col": value_col,
        "group_1": group_order[0],
        "group_2": group_order[1],
        "n_1": len(g1),
        "n_2": len(g2),
        "median_1": float(g1.median()),
        "median_2": float(g2.median()),
        "u_stat": float(stat),
        "p_value": float(p),
    }


# Spearman correlations
spearman_results = []

for x_col, x_label in [
    ("genai_general_frequency_num", "General GenAI usage frequency"),
    ("genai_se_frequency_num", "SE-specific GenAI usage frequency"),
    ("n_genai_tools", "Number of GenAI tools used"),
    ("n_genai_activities", "Number of GenAI-supported activities"),
]:
    for y_col, y_label in [
        ("perception_index", "Perception index"),
        ("role_disruption_index", "Role disruption index"),
        ("automation_revision_needed", "Expected need for revision of automated outputs"),
        ("automation_speed_up", "Belief that automation will speed development"),
    ]:
        sub = analysis_df[[x_col, y_col]].dropna()
        rho, p = spearmanr(sub[x_col], sub[y_col])
        spearman_results.append({
            "x": x_label,
            "y": y_label,
            "n": len(sub),
            "spearman_rho": rho,
            "p_value": p
        })

spearman_results = pd.DataFrame(spearman_results).sort_values(["p_value", "spearman_rho"], ascending=[True, False])
display(spearman_results)


# Two-group comparisons using Mann-Whitney U
mw_results = []

comparisons = [
    ("perception_index", "employed_in_software", ["Não", "Sim"]),
    ("perception_index", "prior_project_with_genai_feature", ["Não", "Sim"]),
    ("genai_se_frequency_num", "employed_in_software", ["Não", "Sim"]),
    ("genai_se_frequency_num", "prior_project_with_genai_feature", ["Não", "Sim"]),
    ("role_disruption_index", "prior_project_with_genai_feature", ["Não", "Sim"]),
]

for value_col, group_col, group_order in comparisons:
    mw_results.append(mannwhitney_compare(analysis_df, value_col, group_col, group_order))

mw_results = pd.DataFrame(mw_results).sort_values("p_value")
display(mw_results)


# Kruskal-Wallis: experience groups and semester groups
kw_results = []

for group_col, order in [
    ("software_experience", software_experience_order),
    ("semester", semester_order),
]:
    for value_col in ["perception_index", "genai_se_frequency_num", "role_disruption_index"]:
        groups = []
        labels = []
        for category in order:
            vals = analysis_df.loc[analysis_df[group_col] == category, value_col].dropna()
            if len(vals) > 0:
                groups.append(vals)
                labels.append(category)
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            kw_results.append({
                "group_col": group_col,
                "value_col": value_col,
                "tested_categories": labels,
                "h_stat": float(stat),
                "p_value": float(p)
            })

kw_results = pd.DataFrame(kw_results).sort_values("p_value")
display(kw_results)


# Contingency analyses
contingency_results = []

for row_var, col_var in [
    ("high_se_usage", "prior_project_with_genai_feature"),
    ("high_se_usage", "employed_in_software"),
    ("daily_or_weekly_general_usage", "prior_project_with_genai_feature"),
]:
    table = pd.crosstab(analysis_df[row_var], analysis_df[col_var])
    chi2, p, dof, expected = chi2_contingency(table)
    contingency_results.append({
        "row_var": row_var,
        "col_var": col_var,
        "table_shape": table.shape,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": float(cramers_v(table))
    })
    print(f"Contingency table: {row_var} x {col_var}")
    display(table)

contingency_results = pd.DataFrame(contingency_results).sort_values("p_value")
display(contingency_results)


"""\n## 9. Segmentation of respondents by adoption intensity

A simple exploratory adoption score is created from:
- general GenAI usage frequency
- SE-specific GenAI usage frequency
- number of GenAI tools used
- number of GenAI-supported activities
- prior participation in projects with GenAI functionality

This is not a validated scale; it is only a segmentation device for EDA.\n"""

analysis_df["adoption_score_raw"] = (
    analysis_df["genai_general_frequency_num"].fillna(0)
    + analysis_df["genai_se_frequency_num"].fillna(0)
    + analysis_df["n_genai_tools"].fillna(0)
    + analysis_df["n_genai_activities"].fillna(0)
    + analysis_df["prior_project_with_genai_feature_num"].fillna(0)
)

analysis_df["adoption_segment"] = pd.qcut(
    analysis_df["adoption_score_raw"],
    q=3,
    labels=["Lower adoption", "Moderate adoption", "Higher adoption"],
    duplicates="drop"
)

display(
    analysis_df[["adoption_score_raw", "adoption_segment"]]
    .groupby("adoption_segment")
    .agg(["count", "mean", "min", "max"])
)


# Segment comparison
segment_order = ["Lower adoption", "Moderate adoption", "Higher adoption"]

for metric, title in [
    ("perception_index", "Perception index by adoption segment"),
    ("role_disruption_index", "Role disruption index by adoption segment"),
    ("automation_revision_needed", "Expected revision need by adoption segment"),
    ("automation_speed_up", "Belief in speed-up by adoption segment"),
]:
    means = analysis_df.groupby("adoption_segment")[metric].mean().reindex(segment_order)
    plt.figure(figsize=(8, 4.5))
    plt.bar(means.index.astype(str), means.values)
    for i, v in enumerate(means.values):
        if pd.notna(v):
            plt.text(i, v + 0.03, f"{v:.2f}", ha="center")
    plt.title(title)
    plt.ylabel("Mean score")
    plt.tight_layout()
    plt.show()


"""\n## 10. Focused interpretation cells

The next cells automatically summarize the strongest descriptive patterns from the dataset.\n"""

# Top descriptive findings (programmatic summary)
summary_lines = []

summary_lines.append(f"- Sample size: **{len(analysis_df)} respondents**.")
summary_lines.append(
    f"- General GenAI use is common: **{analysis_df['daily_or_weekly_general_usage'].mean() * 100:.1f}%** report using it weekly or daily."
)
summary_lines.append(
    f"- SE-specific GenAI use is also substantial: **{analysis_df['high_se_usage'].mean() * 100:.1f}%** report weekly or daily use in software-engineering projects."
)
summary_lines.append(
    f"- Prior experience with GenAI functionality in projects: **{analysis_df['prior_project_with_genai_feature_num'].mean() * 100:.1f}%**."
)

top_tools = multiselect_counts(analysis_df["genai_tools_last_12m"]).head(3)
summary_lines.append(
    "- Most selected GenAI tools: " + ", ".join([f"**{idx} ({val})**" for idx, val in top_tools.items()]) + "."
)

top_activities = multiselect_counts(analysis_df["genai_activities"]).head(4)
summary_lines.append(
    "- Most common GenAI-supported activities: " + ", ".join([f"**{idx} ({val})**" for idx, val in top_activities.items()]) + "."
)

highest_perception = mean_item_scores.sort_values(ascending=False).head(3)
lowest_perception = mean_item_scores.sort_values(ascending=True).head(3)

summary_lines.append(
    "- Highest-agreement perception items: " + ", ".join([f"**{idx} ({val:.2f})**" for idx, val in highest_perception.items()]) + "."
)
summary_lines.append(
    "- Lowest-agreement perception items: " + ", ".join([f"**{idx} ({val:.2f})**" for idx, val in lowest_perception.items()]) + "."
)

most_disrupted_roles = role_means.sort_values(ascending=False).head(3)
least_disrupted_roles = role_means.sort_values(ascending=True).head(3)

summary_lines.append(
    "- Roles perceived as most impacted by GenAI: " + ", ".join([f"**{idx} ({val:.2f})**" for idx, val in most_disrupted_roles.items()]) + "."
)
summary_lines.append(
    "- Roles perceived as least impacted by GenAI: " + ", ".join([f"**{idx} ({val:.2f})**" for idx, val in least_disrupted_roles.items()]) + "."
)

display(Markdown("\n".join(summary_lines)))


# Strongest exploratory statistical signals
top_spearman = spearman_results.head(5).copy()
top_mw = mw_results.head(5).copy()
top_ct = contingency_results.head(5).copy()

display(Markdown("### Strongest exploratory associations detected"))
display(top_spearman)

display(Markdown("### Strongest two-group contrasts detected"))
display(top_mw)

display(Markdown("### Strongest contingency relationships detected"))
display(top_ct)


"""\n## 11. Recommended analytical reading of this dataset

### What this dataset supports well
- descriptive profiling of students/respondents
- adoption mapping
- attitude/perception mapping
- exploratory contrasts between small subgroups
- practical insight generation for curriculum/project design

### What this dataset does **not** support strongly
- causal claims
- population inference with high confidence
- psychometric validation of the composite indices
- fine-grained subgroup analysis for small categories (for example, rare roles)

### Suggested substantive reading
If the strongest patterns hold after reproduction, the data most likely supports a narrative such as:
1. generative AI is already normalized in the respondents' everyday workflow
2. adoption in software engineering is substantial but not uniformly daily
3. perceived value is strongest around usefulness, productivity, and workflow change
4. respondents may be less convinced that GenAI directly improves software quality or that they are fully confident integrating AI-generated code
5. perceived disruption is not uniform across software roles\n"""

"""\n## 12. Optional export of cleaned data and compact analysis tables\n"""

# Export cleaned data and key summary tables for reuse
analysis_df.to_csv(OUTPUT_DIR / "cleaned_anonymized_data.csv", index=False)
schema.to_csv(OUTPUT_DIR / "schema_audit.csv", index=False)
missing_report.to_csv(OUTPUT_DIR / "missing_report.csv", index=False)
spearman_results.to_csv(OUTPUT_DIR / "spearman_results.csv", index=False)
mw_results.to_csv(OUTPUT_DIR / "mannwhitney_results.csv", index=False)
kw_results.to_csv(OUTPUT_DIR / "kruskal_results.csv", index=False)
contingency_results.to_csv(OUTPUT_DIR / "contingency_results.csv", index=False)
mean_item_scores.to_csv(OUTPUT_DIR / "mean_item_scores.csv", header=["mean_score"])
role_means.to_csv(OUTPUT_DIR / "role_means.csv", header=["mean_impact_score"])

print(f"Saved outputs to: {OUTPUT_DIR.resolve()}")


"""\n## 13. Closing note

This notebook is intentionally designed to be:
- transparent in its cleaning steps
- reproducible from the raw CSV
- rich in descriptive visualization
- cautious in statistical interpretation

You can extend it further with:
- bootstrap intervals
- item reliability checks
- subgroup dashboards
- regression models tailored to specific research questions\n"""
