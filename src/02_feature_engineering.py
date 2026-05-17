"""
02_feature_engineering.py — 数据清洗与特征工程
================================================
运行方式：python src/02_feature_engineering.py
输出：data/processed.csv（清洗后的数据）
"""

import pandas as pd
import numpy as np
import os


# ──────────────────────────────────────────
# 1. 数据清洗
# ──────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：
    - 删除明显异常值
    - 填充缺失值
    """
    df = df.copy()
    original_len = len(df)

    # 1.1 删除 age 异常值（age <= 0）
    df = df[df["age"] > 0]

    # 1.2 删除 RevolvingUtilizationOfUnsecuredLines 极端值（信用使用率 > 1 的极大值）
    df = df[df["RevolvingUtilizationOfUnsecuredLines"] <= 1.5]

    # 1.3 填充缺失值
    #   MonthlyIncome：用中位数填充
    monthly_income_median = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(monthly_income_median)

    #   NumberOfDependents：用中位数填充
    dependents_median = df["NumberOfDependents"].median()
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(dependents_median)

    cleaned_len = len(df)
    print(f"[清洗] 删除异常行 {original_len - cleaned_len} 行，剩余 {cleaned_len} 行")
    print(f"[清洗] MonthlyIncome 中位数填充：{monthly_income_median:.0f}")
    print(f"[清洗] NumberOfDependents 中位数填充：{dependents_median:.0f}")
    print(f"[清洗] 剩余缺失值：\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ──────────────────────────────────────────
# 2. 特征工程
# ──────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：
    - 构造衍生特征
    - 对数变换（处理右偏分布）
    - 年龄分段
    """
    df = df.copy()

    # 2.1 债务收入比（月债务 / 月收入）
    # DebtRatio 已经是这个含义，但收入为0时会产生无穷，这里做保护性计算
    df["DebtToIncome"] = df["DebtRatio"] * df["MonthlyIncome"] / (df["MonthlyIncome"] + 1)

    # 2.2 月收入对数（消除右偏）
    df["LogMonthlyIncome"] = np.log1p(df["MonthlyIncome"])

    # 2.3 信用使用率对数
    df["LogRevolvingUtil"] = np.log1p(df["RevolvingUtilizationOfUnsecuredLines"])

    # 2.4 逾期总次数（汇总所有逾期相关特征）
    overdue_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    df["TotalOverdueTimes"] = df[overdue_cols].sum(axis=1)

    # 2.5 是否有过逾期（二值特征）
    df["HasOverdue"] = (df["TotalOverdueTimes"] > 0).astype(int)

    # 2.6 年龄分段
    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # 2.7 每个信用账户的债务率（信用账户数量保护）
    df["DebtPerAccount"] = df["DebtRatio"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)

    print(f"[特征工程] 新增特征 7 个，当前总特征数：{df.shape[1]}")
    print(f"[特征工程] 新特征列表：DebtToIncome, LogMonthlyIncome, LogRevolvingUtil, "
          f"TotalOverdueTimes, HasOverdue, AgeGroup, DebtPerAccount")
    return df


# ──────────────────────────────────────────
# 3. 变量分箱（WOE 编码，信贷项目标准做法）
# ──────────────────────────────────────────

def calc_woe_iv(df: pd.DataFrame, feature: str, target: str = "SeriousDlqin2yrs",
                bins: int = 10) -> pd.DataFrame:
    """
    计算单个特征的 WOE（Weight of Evidence）和 IV（Information Value）
    WOE_i = ln(好客户占比_i / 坏客户占比_i)
    IV = Σ (好客户占比_i - 坏客户占比_i) * WOE_i
    """
    df_tmp = df[[feature, target]].copy().dropna()

    # 等频分箱
    try:
        df_tmp["bin"] = pd.qcut(df_tmp[feature], q=bins, duplicates="drop")
    except Exception:
        df_tmp["bin"] = pd.cut(df_tmp[feature], bins=bins, duplicates="drop")

    grouped = df_tmp.groupby("bin", observed=False)[target].agg(["sum", "count"])
    grouped.columns = ["bad", "total"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()

    grouped["bad_rate"] = grouped["bad"] / total_bad
    grouped["good_rate"] = grouped["good"] / total_good

    # 防止除零
    grouped["bad_rate"] = grouped["bad_rate"].replace(0, 0.0001)
    grouped["good_rate"] = grouped["good_rate"].replace(0, 0.0001)

    grouped["WOE"] = np.log(grouped["good_rate"] / grouped["bad_rate"])
    grouped["IV"] = (grouped["good_rate"] - grouped["bad_rate"]) * grouped["WOE"]

    return grouped


def compute_all_iv(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有特征的 IV 值，并打印排名"""
    features = [c for c in df.columns if c != "SeriousDlqin2yrs"]
    iv_results = []

    for feat in features:
        try:
            woe_df = calc_woe_iv(df, feat)
            iv = woe_df["IV"].sum()
            iv_results.append({"feature": feat, "IV": round(iv, 4)})
        except Exception as e:
            iv_results.append({"feature": feat, "IV": 0.0})

    iv_df = pd.DataFrame(iv_results).sort_values("IV", ascending=False)

    print("\n[WOE/IV] 特征 IV 值排名（IV > 0.02 有预测价值）：")
    print(iv_df.to_string(index=False))

    # IV 判断标准
    print("\nIV 参考标准：")
    print("  < 0.02  → 预测力弱（可考虑删除）")
    print("  0.02~0.1 → 弱预测力")
    print("  0.1~0.3  → 中等预测力")
    print("  > 0.3    → 强预测力")

    return iv_df


# ──────────────────────────────────────────
# 4. 主流程
# ──────────────────────────────────────────

def get_feature_columns() -> list:
    """返回建模使用的特征列表"""
    return [
        # 原始特征
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
        # 衍生特征
        "DebtToIncome",
        "LogMonthlyIncome",
        "LogRevolvingUtil",
        "TotalOverdueTimes",
        "HasOverdue",
        "AgeGroup",
        "DebtPerAccount",
    ]


def run_feature_engineering(input_path: str = "data/cs-training.csv",
                             output_path: str = "data/processed.csv"):
    df = pd.read_csv(input_path, index_col=0)
    print(f"[加载] {len(df)} 行原始数据")

    df = clean_data(df)
    df = engineer_features(df)

    # 计算 IV
    compute_all_iv(df)

    # 保存
    df.to_csv(output_path, index=False)
    print(f"\n[完成] 处理后数据已保存：{output_path}（{df.shape[0]} 行 × {df.shape[1]} 列）")
    return df


if __name__ == "__main__":
    run_feature_engineering()
