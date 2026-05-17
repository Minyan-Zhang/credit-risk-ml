"""
app.py — 信贷违约风险评估系统 Web 应用
运行方式：streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score,
    f1_score, roc_curve, confusion_matrix, classification_report
)

# ─────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────
st.set_page_config(
    page_title="信贷违约风险评估系统",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42
COLORS = {"LogisticRegression": "#4C72B0",
          "RandomForest":       "#DD8452",
          "GradientBoosting":   "#55A868"}

FEAT_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit Util Rate",
    "age":                                  "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 Days Late",
    "DebtRatio":                            "Debt Ratio",
    "MonthlyIncome":                        "Monthly Income",
    "NumberOfOpenCreditLinesAndLoans":      "Open Credit Lines",
    "NumberOfTimes90DaysLate":              "90+ Days Late",
    "NumberRealEstateLoansOrLines":         "Real Estate Loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 Days Late",
    "NumberOfDependents":                   "Dependents",
    "DebtToIncome":                         "Debt-to-Income",
    "LogMonthlyIncome":                     "Log Income",
    "LogRevolvingUtil":                     "Log Credit Util",
    "TotalOverdueTimes":                    "Total Overdue",
    "HasOverdue":                           "Has Overdue",
    "AgeGroup":                             "Age Group",
    "DebtPerAccount":                       "Debt per Account",
}


# ─────────────────────────────────────────
# 数据工具函数
# ─────────────────────────────────────────

@st.cache_data
def generate_demo_data(n=30000, seed=42):
    rng = np.random.default_rng(seed)
    age      = rng.integers(21, 80, n).astype(float)
    income   = np.exp(rng.normal(8.5, 0.8, n))
    income[rng.random(n) < 0.08] = np.nan
    debt_ratio  = rng.beta(1.5, 5, n) * 2
    revolving   = np.clip(rng.beta(1.2, 3, n), 0, 1.0)
    late_30_59  = rng.negative_binomial(1, 0.85, n)
    late_60_89  = rng.negative_binomial(1, 0.92, n)
    late_90     = rng.negative_binomial(1, 0.94, n)
    open_credit = rng.poisson(8, n)
    real_estate = rng.poisson(1, n)
    dependents  = rng.poisson(0.8, n).astype(float)
    dependents[rng.random(n) < 0.025] = np.nan
    log_odds = (-4.5
        + 1.8 * (late_30_59 + late_60_89 * 1.5 + late_90 * 2.0)
        + 1.2 * revolving + 0.8 * debt_ratio
        - 0.5 * np.log1p(np.nan_to_num(income, nan=4000) / 5000)
        - 0.03 * (age - 40) + rng.normal(0, 0.5, n))
    prob = 1 / (1 + np.exp(-log_odds))
    y = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        TARGET: y,
        "RevolvingUtilizationOfUnsecuredLines": revolving,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": open_credit,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": dependents,
    })


def clean_and_engineer(df):
    df = df.copy()
    df = df[df["age"] > 0]
    df = df[df["RevolvingUtilizationOfUnsecuredLines"] <= 1.5]
    df["MonthlyIncome"]      = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())
    df["DebtToIncome"]     = df["DebtRatio"] * df["MonthlyIncome"] / (df["MonthlyIncome"] + 1)
    df["LogMonthlyIncome"] = np.log1p(df["MonthlyIncome"])
    df["LogRevolvingUtil"] = np.log1p(df["RevolvingUtilizationOfUnsecuredLines"])
    df["TotalOverdueTimes"] = (df["NumberOfTime30-59DaysPastDueNotWorse"]
                              + df["NumberOfTime60-89DaysPastDueNotWorse"]
                              + df["NumberOfTimes90DaysLate"])
    df["HasOverdue"]  = (df["TotalOverdueTimes"] > 0).astype(int)
    df["AgeGroup"]    = pd.cut(df["age"], bins=[0,30,40,50,60,100],
                               labels=[0,1,2,3,4]).astype(int)
    df["DebtPerAccount"] = df["DebtRatio"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
    return df


def get_feature_cols(df):
    candidates = list(FEAT_LABELS.keys())
    return [f for f in candidates if f in df.columns]


def find_threshold(y_true, y_prob):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        yp = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, yp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


@st.cache_resource
def train_models(df_processed):
    features = get_feature_cols(df_processed)
    X = df_processed[features]
    y = df_processed[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models_def = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced",
                               random_state=RANDOM_STATE),
            X_train_sc, X_test_sc),
        "RandomForest": (
            RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=30,
                                   class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            X_train, X_test),
        "GradientBoosting": (
            GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, random_state=RANDOM_STATE),
            X_train, X_test),
    }

    results = {}
    for name, (model, Xtr, Xte) in models_def.items():
        model.fit(Xtr, y_train)
        y_prob = model.predict_proba(Xte)[:, 1]
        thr    = find_threshold(y_test, y_prob)
        y_pred = (y_prob >= thr).astype(int)
        results[name] = dict(
            model=model, scaler=scaler if name=="LogisticRegression" else None,
            y_prob=y_prob, y_pred=y_pred, threshold=thr,
            AUC=round(roc_auc_score(y_test, y_prob), 4),
            Recall=round(recall_score(y_test, y_pred), 4),
            Precision=round(precision_score(y_test, y_pred, zero_division=0), 4),
            F1=round(f1_score(y_test, y_pred, zero_division=0), 4),
        )
    return results, X_test, y_test, features, scaler


# ─────────────────────────────────────────
# 图表函数
# ─────────────────────────────────────────

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_target_dist(df):
    vc = df[TARGET].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(["Normal (0)", "Default (1)"], vc.values, color=["#4C72B0","#DD8452"], edgecolor="white", width=0.5)
    axes[0].set_title("Count", fontsize=11)
    for i, v in enumerate(vc.values):
        axes[0].text(i, v*1.02, f"{v:,}", ha="center", fontsize=10)
    axes[1].pie(vc.values, labels=["Normal (0)","Default (1)"], autopct="%1.1f%%",
                colors=["#4C72B0","#DD8452"], startangle=90,
                wedgeprops={"edgecolor":"white","linewidth":1.5})
    axes[1].set_title("Proportion", fontsize=11)
    fig.suptitle("Target Variable Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feat_dist(df):
    feats = [c for c in df.columns if c != TARGET][:10]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()
    for i, feat in enumerate(feats):
        data = df[feat].dropna()
        data = data[data <= data.quantile(0.99)]
        axes[i].hist(data, bins=35, color=plt.cm.tab10(i/10), edgecolor="white", alpha=0.85)
        axes[i].set_title(FEAT_LABELS.get(feat, feat), fontsize=8)
        axes[i].tick_params(labelsize=7)
    fig.suptitle("Feature Distributions (clipped at 99th pct)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_corr(df):
    fig, ax = plt.subplots(figsize=(11, 8))
    corr = df.fillna(df.median()).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    short_cols = {c: FEAT_LABELS.get(c, c)[:12] for c in corr.columns}
    corr_display = corr.rename(columns=short_cols, index=short_cols)
    sns.heatmap(corr_display, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.4, annot_kws={"size": 6},
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, lw=2.5, color=COLORS[name],
                label=f"{name}  AUC={res['AUC']:.4f}")
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random  AUC=0.50")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_cm(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\n阈值={res['threshold']:.2f}", fontsize=9, fontweight="bold")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Normal","Default"]); ax.set_yticklabels(["Normal","Default"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if cm[i,j]>cm.max()/2 else "black")
        plt.colorbar(im, ax=ax)
    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_metrics(results):
    metrics = ["AUC","Recall","Precision","F1"]
    names   = list(results.keys())
    x, w    = np.arange(len(metrics)), 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x+i*w, vals, w, label=name, color=list(COLORS.values())[i],
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.007,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x+w); ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_importance(results, features, model_name):
    model = results[model_name]["model"]
    imp   = model.feature_importances_
    labels = [FEAT_LABELS.get(f, f) for f in features]
    order  = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = imp[order] / imp[order].max()
    cmap = plt.cm.RdYlGn if model_name == "GradientBoosting" else plt.cm.Blues
    ax.barh([labels[i] for i in order], imp[order],
            color=cmap(norm * 0.7 + 0.3), edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"{model_name} 特征重要性", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_lr_coef(results, features):
    model = results["LogisticRegression"]["model"]
    coef  = model.coef_[0]
    labels = [FEAT_LABELS.get(f, f) for f in features]
    order  = np.argsort(coef)
    colors = ["#DD8452" if coef[i]>0 else "#4C72B0" for i in order]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([labels[i] for i in order], coef[order], color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Coefficient (positive = higher default risk)", fontsize=10)
    ax.set_title("Logistic Regression Coefficients", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[mpatches.Patch(color="#DD8452",label="Risk Increase"),
                        mpatches.Patch(color="#4C72B0",label="Risk Decrease")], fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# 单用户预测
# ─────────────────────────────────────────

def predict_single(user_input, results, features, scaler):
    best_model_name = max(results, key=lambda n: results[n]["AUC"])
    res = results[best_model_name]
    model = res["model"]

    input_df = pd.DataFrame([user_input])
    # 补全衍生特征
    input_df["DebtToIncome"]     = input_df["DebtRatio"] * input_df["MonthlyIncome"] / (input_df["MonthlyIncome"]+1)
    input_df["LogMonthlyIncome"] = np.log1p(input_df["MonthlyIncome"])
    input_df["LogRevolvingUtil"] = np.log1p(input_df["RevolvingUtilizationOfUnsecuredLines"])
    input_df["TotalOverdueTimes"] = (input_df["NumberOfTime30-59DaysPastDueNotWorse"]
                                    + input_df["NumberOfTime60-89DaysPastDueNotWorse"]
                                    + input_df["NumberOfTimes90DaysLate"])
    input_df["HasOverdue"]  = (input_df["TotalOverdueTimes"] > 0).astype(int)
    input_df["AgeGroup"]    = pd.cut(input_df["age"], bins=[0,30,40,50,60,100],
                                     labels=[0,1,2,3,4]).astype(int)
    input_df["DebtPerAccount"] = input_df["DebtRatio"] / (input_df["NumberOfOpenCreditLinesAndLoans"]+1)

    X = input_df[features]
    if best_model_name == "LogisticRegression":
        X = scaler.transform(X)

    prob = model.predict_proba(X)[0][1]
    label = 1 if prob >= res["threshold"] else 0
    return prob, label, best_model_name


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────

with st.sidebar:
    st.title("💳 信贷违约风险评估")
    st.markdown("---")
    page = st.radio("📌 功能导航", [
        "🏠 项目介绍",
        "📊 数据探索 (EDA)",
        "⚙️ 特征工程",
        "🤖 模型训练与评估",
        "📈 特征重要性",
        "🔍 单用户风险预测",
    ])
    st.markdown("---")
    st.caption("数据集：Kaggle Give Me Some Credit")
    st.caption("模型：LR · RF · GradientBoosting")


# ─────────────────────────────────────────
# 加载数据（只加载一次）
# ─────────────────────────────────────────

with st.spinner("初始化数据中..."):
    df_raw       = generate_demo_data()
    df_processed = clean_and_engineer(df_raw)

    if "models_trained" not in st.session_state:
        with st.spinner("训练模型中，请稍候..."):
            results, X_test, y_test, features, scaler = train_models(df_processed)
            st.session_state["models_trained"] = True
            st.session_state["results"]  = results
            st.session_state["X_test"]   = X_test
            st.session_state["y_test"]   = y_test
            st.session_state["features"] = features
            st.session_state["scaler"]   = scaler
    else:
        results  = st.session_state["results"]
        X_test   = st.session_state["X_test"]
        y_test   = st.session_state["y_test"]
        features = st.session_state["features"]
        scaler   = st.session_state["scaler"]


# ═════════════════════════════════════════
# 各页面内容
# ═════════════════════════════════════════

# ── 项目介绍 ──────────────────────────────
if page == "🏠 项目介绍":
    st.title("💳 基于机器学习的信贷违约风险评估系统")
    st.markdown("""
    ### 项目背景
    信贷违约风险评估是金融机构核心业务之一。本系统基于用户历史信用数据，
    通过机器学习模型**预测用户未来2年内是否发生严重违约（二分类任务）**。

    ### 数据集
    使用 [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) 数据集
    （演示模式使用同结构模拟数据，替换为真实数据只需上传 CSV）。
    """)

    col1, col2, col3, col4 = st.columns(4)
    vc = df_raw[TARGET].value_counts()
    col1.metric("总样本数", f"{len(df_raw):,}")
    col2.metric("正常用户", f"{vc[0]:,}")
    col3.metric("违约用户", f"{vc[1]:,}")
    col4.metric("违约率", f"{vc[1]/len(df_raw):.1%}")

    st.markdown("---")
    st.markdown("""
    ### 技术方案
    | 步骤 | 内容 |
    |------|------|
    | 数据清洗 | 异常值删除、缺失值中位数填充 |
    | 特征工程 | 构造7个衍生特征、WOE/IV计算 |
    | 不平衡处理 | class_weight='balanced'、调整分类阈值 |
    | 建模 | Logistic Regression · Random Forest · Gradient Boosting |
    | 评估 | AUC · Recall · Precision · F1 |

    ### 模型结果预览
    """)
    summary = pd.DataFrame([
        {"模型": n, "AUC": r["AUC"], "Recall": r["Recall"],
         "Precision": r["Precision"], "F1": r["F1"]}
        for n, r in results.items()
    ]).set_index("模型")
    st.dataframe(summary.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)


# ── EDA ──────────────────────────────────
elif page == "📊 数据探索 (EDA)":
    st.title("📊 探索性数据分析")

    tab1, tab2, tab3, tab4 = st.tabs(["Target Variable Distribution", "特征分布", "相关性热力图", "分组箱线图"])

    with tab1:
        st.subheader("Target Variable Distribution")
        st.pyplot(plot_target_dist(df_raw))
        vc = df_raw[TARGET].value_counts()
        st.info(f"⚠️ 样本严重不平衡：正常 vs 违约 ≈ **{vc[0]//vc[1]}:1**，需要特殊处理。")

    with tab2:
        st.subheader("原始特征分布")
        st.pyplot(plot_feat_dist(df_raw))

    with tab3:
        st.subheader("Feature Correlation Heatmap")
        st.pyplot(plot_corr(df_raw))

    with tab4:
        st.subheader("按违约/正常分组的特征分布")
        feat_choice = st.selectbox("选择特征", options=[
            "RevolvingUtilizationOfUnsecuredLines","age","DebtRatio",
            "MonthlyIncome","NumberOfTimes90DaysLate"
        ], format_func=lambda x: FEAT_LABELS.get(x, x))
        fig, ax = plt.subplots(figsize=(7, 4))
        g0 = df_raw[df_raw[TARGET]==0][feat_choice].dropna()
        g1 = df_raw[df_raw[TARGET]==1][feat_choice].dropna()
        p99 = df_raw[feat_choice].quantile(0.99)
        bp = ax.boxplot([g0[g0<=p99], g1[g1<=p99]], labels=["Normal (0)","Default (1)"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#4C72B0"); bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#DD8452"); bp["boxes"][1].set_alpha(0.6)
        ax.set_title(FEAT_LABELS.get(feat_choice, feat_choice), fontsize=12)
        st.pyplot(fig); plt.close()


# ── 特征工程 ────────────────────────────
elif page == "⚙️ 特征工程":
    st.title("⚙️ 数据清洗与特征工程")
    st.markdown("### 清洗步骤")
    st.markdown("""
    - **删除** `age <= 0` 的异常记录
    - **删除** `信用卡使用率 > 1.5` 的极端值
    - **月收入** 缺失值用中位数填充
    - **赡养人数** 缺失值用中位数填充
    """)

    st.markdown("### 衍生特征（新增7个）")
    st.dataframe(pd.DataFrame({
        "新特征": ["DebtToIncome","LogMonthlyIncome","LogRevolvingUtil",
                   "TotalOverdueTimes","HasOverdue","AgeGroup","DebtPerAccount"],
        "含义": ["月债务金额","月收入(对数，消右偏)","信用使用率(对数)",
                  "所有逾期次数汇总","是否有过逾期(0/1)","年龄分段(0-4)","每账户债务率"],
        "业务意义": ["衡量实际债务压力","消除收入的右偏分布","消除使用率的右偏分布",
                     "综合衡量历史违约行为","二值化逾期信号","捕捉年龄段效应","单账户负载水平"]
    }), use_container_width=True, hide_index=True)

    st.markdown("### WOE / IV 特征预测力")
    feats_for_iv = [c for c in df_processed.columns if c != TARGET]
    iv_list = []
    for feat in feats_for_iv:
        try:
            tmp = df_processed[[feat, TARGET]].dropna()
            tmp["bin"] = pd.qcut(tmp[feat], q=8, duplicates="drop")
            g = tmp.groupby("bin", observed=False)[TARGET].agg(["sum","count"])
            g["bad"] = g["sum"]; g["good"] = g["count"] - g["sum"]
            tb = g["bad"].sum(); tg = g["good"].sum()
            g["br"] = (g["bad"]/tb).replace(0,1e-4)
            g["gr"] = (g["good"]/tg).replace(0,1e-4)
            g["WOE"] = np.log(g["gr"]/g["br"])
            iv = ((g["gr"]-g["br"])*g["WOE"]).sum()
        except:
            iv = 0.0
        iv_list.append({"特征": FEAT_LABELS.get(feat, feat), "IV值": round(iv,4),
                         "预测力": "强 🟢" if iv>0.3 else ("中 🟡" if iv>0.1 else ("弱 🟠" if iv>0.02 else "无效 🔴"))})
    iv_df = pd.DataFrame(iv_list).sort_values("IV值", ascending=False)
    st.dataframe(iv_df, use_container_width=True, hide_index=True)


# ── 模型训练与评估 ────────────────────────
elif page == "🤖 模型训练与评估":
    st.title("🤖 模型训练与评估")

    tab1, tab2, tab3 = st.tabs(["ROC 曲线", "Confusion Matrices", "指标对比"])

    with tab1:
        st.subheader("ROC 曲线")
        st.pyplot(plot_roc(results, y_test))
        st.success("💡 AUC 越接近 1.0 越好，代表模型区分违约/正常用户的能力越强")

    with tab2:
        st.subheader("Confusion Matrices")
        st.pyplot(plot_cm(results, y_test))
        st.caption("TN=正确识别正常 | FP=误报违约 | FN=漏报违约 | TP=正确识别违约")

    with tab3:
        st.subheader("各模型指标对比")
        st.pyplot(plot_metrics(results))
        summary = pd.DataFrame([
            {"模型":n, "AUC":r["AUC"], "Recall":r["Recall"],
             "Precision":r["Precision"], "F1":r["F1"], "最优阈值":round(r["threshold"],2)}
            for n, r in results.items()
        ]).set_index("模型")
        st.dataframe(summary.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)


# ── 特征重要性 ────────────────────────────
elif page == "📈 特征重要性":
    st.title("📈 特征重要性分析")

    tab1, tab2, tab3 = st.tabs(["GradientBoosting", "RandomForest", "LR 系数"])

    with tab1:
        st.pyplot(plot_importance(results, features, "GradientBoosting"))
    with tab2:
        st.pyplot(plot_importance(results, features, "RandomForest"))
    with tab3:
        st.pyplot(plot_lr_coef(results, features))

    st.markdown("---")
    st.markdown("""
    ### 🔑 高风险用户关键因素
    | 因素 | 风险逻辑 |
    |------|----------|
    | **逾期总次数** | 历史逾期越多，未来违约概率显著升高 |
    | **90天+严重逾期** | 严重违约的最强预测信号 |
    | **信用卡使用率** | 使用率>80% 说明资金极度紧张 |
    | **年龄** | 年轻用户（<30岁）信用历史短，风险更高 |
    | **负债率** | 负债率越高，还款压力越大 |
    """)


# ── 单用户预测 ────────────────────────────
elif page == "🔍 单用户风险预测":
    st.title("🔍 单用户违约风险预测")
    st.markdown("输入用户信息，系统实时给出违约概率和风险等级。")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age        = st.slider("年龄", 18, 80, 35)
        income     = st.number_input("月收入（元）", 0, 100000, 8000, step=500)
        debt_ratio = st.slider("负债率（月债务/月收入）", 0.0, 3.0, 0.3, 0.01)
        revolving  = st.slider("信用卡使用率（0~1）", 0.0, 1.0, 0.3, 0.01)
        open_credit= st.slider("开放信用账户数", 0, 30, 8)

    with col2:
        late_30_59 = st.number_input("30-59天逾期次数", 0, 20, 0)
        late_60_89 = st.number_input("60-89天逾期次数", 0, 20, 0)
        late_90    = st.number_input("90天+逾期次数",   0, 20, 0)
        real_estate= st.slider("房产贷款数", 0, 10, 1)
        dependents = st.slider("赡养人数", 0, 10, 0)

    if st.button("🚀 开始预测", use_container_width=True, type="primary"):
        user_input = {
            "RevolvingUtilizationOfUnsecuredLines": revolving,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": open_credit,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "NumberOfDependents": dependents,
        }
        prob, label, model_name = predict_single(user_input, results, features, scaler)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Default Probability", f"{prob:.1%}")
        c2.metric("预测结果", "⚠️ 高风险违约" if label==1 else "✅ 低风险正常")
        c3.metric("使用模型", model_name)

        if prob < 0.3:
            risk_color, risk_label, advice = "🟢", "低风险", "用户信用状况良好，建议正常审批。"
        elif prob < 0.6:
            risk_color, risk_label, advice = "🟡", "中风险", "存在一定风险，建议核实收入和逾期记录后谨慎审批。"
        else:
            risk_color, risk_label, advice = "🔴", "高风险", "违约风险较高，建议拒绝申请或降低授信额度。"

        st.markdown(f"""
        ### {risk_color} 风险评级：{risk_label}

        **建议**：{advice}
        """)

        fig, ax = plt.subplots(figsize=(7, 1.5))
        ax.barh(["Default Probability"], [prob], color="#DD8452" if label==1 else "#4C72B0",
                height=0.4, alpha=0.8)
        ax.barh(["Default Probability"], [1-prob], left=[prob], color="#e0e0e0", height=0.4)
        ax.axvline(results[model_name]["threshold"], color="red", lw=2, linestyle="--",
                   label=f"阈值={results[model_name]['threshold']:.2f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Default Probability")
        ax.text(prob/2, 0, f"{prob:.1%}", ha="center", va="center",
                fontweight="bold", fontsize=12, color="white")
        ax.legend(fontsize=9)
        ax.set_title("Default Probability Visualization", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
