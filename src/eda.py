"""
eda.py — 供 main.py 导入的 EDA 函数（与 01_eda.py 内容相同）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str = "data/cs-training.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到数据文件：{path}\n"
            "请先从 Kaggle 下载：https://www.kaggle.com/c/GiveMeSomeCredit\n"
            "将 cs-training.csv 放到 data/ 目录下。"
        )
    df = pd.read_csv(path, index_col=0)
    print(f"[数据加载] 共 {df.shape[0]} 行，{df.shape[1]} 列")
    return df


def basic_info(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("【1】数据基本信息")
    print("=" * 60)
    print(df.dtypes)
    print("\n【2】前5行数据")
    print(df.head())
    print("\n【3】描述性统计")
    print(df.describe().T.to_string())
    print("\n【4】缺失值统计")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"缺失数量": missing, "缺失比例(%)": missing_pct})
    print(missing_df[missing_df["缺失数量"] > 0])
    print("\n【5】目标变量分布")
    vc = df["SeriousDlqin2yrs"].value_counts()
    print(vc)
    print(f"  正负样本比例 = 1 : {vc[0] / vc[1]:.1f}")


def plot_target_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Target Variable Distribution (SeriousDlqin2yrs)", fontsize=13)
    vc = df["SeriousDlqin2yrs"].value_counts()
    axes[0].bar(["Normal (0)", "Default (1)"], vc.values, color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Count")
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + 200, str(v), ha="center", fontsize=11)
    axes[1].pie(vc.values, labels=["Normal (0)", "Default (1)"],
                autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"], startangle=90)
    axes[1].set_title("Proportion")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_target_distribution.png", dpi=150)
    plt.close()
    print(f"[图表] 保存：{OUTPUT_DIR}/01_target_distribution.png")


def plot_feature_distributions(df: pd.DataFrame):
    features = [c for c in df.columns if c != "SeriousDlqin2yrs"]
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        data = df[feat].dropna()
        p99 = data.quantile(0.99)
        data_clipped = data[data <= p99]
        axes[i].hist(data_clipped, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[i].set_title(feat, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature Distributions (clipped at 99th percentile)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表] 保存：{OUTPUT_DIR}/02_feature_distributions.png")


def plot_correlation(df: pd.DataFrame):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5, annot_kws={"size": 8})
    plt.title("Feature Correlation Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"[图表] 保存：{OUTPUT_DIR}/03_correlation_heatmap.png")


def plot_boxplot_by_target(df: pd.DataFrame):
    features = ["RevolvingUtilizationOfUnsecuredLines", "age",
                "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans"]
    fig, axes = plt.subplots(1, len(features), figsize=(16, 5))
    for ax, feat in zip(axes, features):
        groups = [df[df["SeriousDlqin2yrs"] == 0][feat].dropna(),
                  df[df["SeriousDlqin2yrs"] == 1][feat].dropna()]
        p99 = df[feat].quantile(0.99)
        groups = [g[g <= p99] for g in groups]
        ax.boxplot(groups, labels=["Normal", "Default"], patch_artist=True,
                   boxprops=dict(facecolor="#4C72B0", alpha=0.6))
        ax.set_title(feat[:20], fontsize=8)
    plt.suptitle("Feature Distribution by Target", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_boxplot_by_target.png", dpi=150)
    plt.close()
    print(f"[图表] 保存：{OUTPUT_DIR}/04_boxplot_by_target.png")
