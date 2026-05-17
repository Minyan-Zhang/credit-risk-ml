"""
feature_importance.py — 供 main.py 导入的特征重要性函数
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

FEATURE_LABELS = {
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


def plot_xgboost_importance(features: list):
    model = joblib.load(f"{MODEL_DIR}/XGBoost.pkl")
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": features,
        "importance": importance,
        "label": [FEATURE_LABELS.get(f, f) for f in features]
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    norm_imp = feat_imp["importance"] / feat_imp["importance"].max()
    bars = ax.barh(feat_imp["label"], feat_imp["importance"],
                   color=plt.cm.RdYlGn(norm_imp))
    ax.set_xlabel("Importance (Gain)", fontsize=12)
    ax.set_title("XGBoost Feature Importance", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, feat_imp["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_xgboost_importance.png", dpi=150)
    plt.close()
    print(f"[图表] 保存：{OUTPUT_DIR}/08_xgboost_importance.png")
    return feat_imp


def plot_random_forest_importance(features: list):
    model = joblib.load(f"{MODEL_DIR}/RandomForest.pkl")
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": features,
        "importance": importance,
        "label": [FEATURE_LABELS.get(f, f) for f in features]
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    norm = feat_imp["importance"] / feat_imp["importance"].max()
    ax.barh(feat_imp["label"], feat_imp["importance"],
            color=plt.cm.Blues(norm * 0.7 + 0.3))
    ax.set_xlabel("Importance (MDI)", fontsize=12)
    ax.set_title("Random Forest Feature Importance", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_rf_importance.png", dpi=150)
    plt.close()
    print(f"[图表] 保存：{OUTPUT_DIR}/09_rf_importance.png")


def plot_logistic_coefficients(features: list):
    model = joblib.load(f"{MODEL_DIR}/LogisticRegression.pkl")
    coef = model.coef_[0]
    feat_imp = pd.DataFrame({
        "feature": features,
        "coef": coef,
        "label": [FEATURE_LABELS.get(f, f) for f in features]
    }).sort_values("coef", ascending=True)

    colors = ["#DD8452" if c > 0 else "#4C72B0" for c in feat_imp["coef"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(feat_imp["label"], feat_imp["coef"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient (positive = higher default risk)", fontsize=11)
    ax.set_title("Logistic Regression Coefficients", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#DD8452", label="Risk Increase"),
                       Patch(facecolor="#4C72B0", label="Risk Decrease")]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_lr_coefficients.png", dpi=150)
    plt.close()
    print(f"[图表] 保存：{OUTPUT_DIR}/10_lr_coefficients.png")


def print_business_insights(xgb_imp: pd.DataFrame):
    top5 = xgb_imp.sort_values("importance", ascending=False).head(5)
    insights = {
        "TotalOverdueTimes":                    "历史逾期次数越多，违约风险显著升高",
        "HasOverdue":                           "有过任何逾期记录的用户，违约概率大幅增加",
        "NumberOfTimes90DaysLate":              "90天以上严重逾期是最强违约信号",
        "RevolvingUtilizationOfUnsecuredLines": "信用卡使用率越高，说明资金紧张，风险越大",
        "LogRevolvingUtil":                     "信用使用率（对数）越高，违约风险越高",
        "age":                                  "年龄较小的用户（尤其<30岁）违约率更高",
        "DebtRatio":                            "负债率越高，还款压力越大，风险越高",
        "MonthlyIncome":                        "收入越低，还款能力越弱",
        "DebtToIncome":                         "债务收入比越高，财务压力越大",
    }
    print("\n" + "=" * 60)
    print("  业务洞察：高风险用户关键因素（XGBoost Top-5）")
    print("=" * 60)
    for _, row in top5.iterrows():
        feat = row["feature"]
        label = row["label"]
        imp = row["importance"]
        insight = insights.get(feat, "该特征对违约预测有重要贡献")
        print(f"\n  [{label}]（重要性：{imp:.4f}）")
        print(f"    → {insight}")
    print("\n" + "=" * 60)
    print("  风险管理建议")
    print("=" * 60)
    print("  1. 重点关注有逾期记录（尤其90天+）的申请人")
    print("  2. 信用卡使用率 > 80% 的用户需要额外审核")
    print("  3. 年轻用户（<30岁）+高负债率 是高风险组合")
    print("  4. 月收入较低且负债率高的用户，建议降低授信额度")


def run_feature_importance(features: list):
    print("[特征重要性分析] 开始...")
    xgb_imp = plot_xgboost_importance(features)
    plot_random_forest_importance(features)
    plot_logistic_coefficients(features)
    print_business_insights(xgb_imp)
    print("\n[特征重要性分析] 完成！")
