"""
feature_engineering_module.py — 供 main.py 导入
"""
import pandas as pd
import numpy as np
import os


def get_feature_columns() -> list:
    return [
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
        "DebtToIncome",
        "LogMonthlyIncome",
        "LogRevolvingUtil",
        "TotalOverdueTimes",
        "HasOverdue",
        "AgeGroup",
        "DebtPerAccount",
    ]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    original_len = len(df)
    df = df[df["age"] > 0]
    df = df[df["RevolvingUtilizationOfUnsecuredLines"] <= 1.5]
    monthly_income_median = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(monthly_income_median)
    dependents_median = df["NumberOfDependents"].median()
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(dependents_median)
    cleaned_len = len(df)
    print(f"[清洗] 删除异常行 {original_len - cleaned_len} 行，剩余 {cleaned_len} 行")
    print(f"[清洗] MonthlyIncome 中位数填充：{monthly_income_median:.0f}")
    print(f"[清洗] NumberOfDependents 中位数填充：{dependents_median:.0f}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DebtToIncome"] = df["DebtRatio"] * df["MonthlyIncome"] / (df["MonthlyIncome"] + 1)
    df["LogMonthlyIncome"] = np.log1p(df["MonthlyIncome"])
    df["LogRevolvingUtil"] = np.log1p(df["RevolvingUtilizationOfUnsecuredLines"])
    overdue_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    df["TotalOverdueTimes"] = df[overdue_cols].sum(axis=1)
    df["HasOverdue"] = (df["TotalOverdueTimes"] > 0).astype(int)
    df["AgeGroup"] = pd.cut(
        df["age"], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4]
    ).astype("object").fillna(-1).astype(int)
    df["DebtPerAccount"] = df["DebtRatio"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
    print(f"[特征工程] 新增特征 7 个，当前总特征数：{df.shape[1]}")
    return df


def calc_iv(df: pd.DataFrame, feature: str, target: str = "SeriousDlqin2yrs", bins: int = 10) -> float:
    df_tmp = df[[feature, target]].copy().dropna()
    try:
        df_tmp["bin"] = pd.qcut(df_tmp[feature], q=bins, duplicates="drop")
    except Exception:
        df_tmp["bin"] = pd.cut(df_tmp[feature], bins=bins, duplicates="drop")
    grouped = df_tmp.groupby("bin", observed=False)[target].agg(["sum", "count"])
    grouped.columns = ["bad", "total"]
    grouped["good"] = grouped["total"] - grouped["bad"]
    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()
    grouped["bad_rate"] = (grouped["bad"] / total_bad).replace(0, 0.0001)
    grouped["good_rate"] = (grouped["good"] / total_good).replace(0, 0.0001)
    grouped["WOE"] = np.log(grouped["good_rate"] / grouped["bad_rate"])
    grouped["IV"] = (grouped["good_rate"] - grouped["bad_rate"]) * grouped["WOE"]
    return grouped["IV"].sum()


def compute_all_iv(df: pd.DataFrame):
    features = [c for c in df.columns if c != "SeriousDlqin2yrs"]
    iv_results = []
    for feat in features:
        try:
            iv = calc_iv(df, feat)
        except Exception:
            iv = 0.0
        iv_results.append({"feature": feat, "IV": round(iv, 4)})
    iv_df = pd.DataFrame(iv_results).sort_values("IV", ascending=False)
    print("\n[WOE/IV] 特征 IV 值排名：")
    print(iv_df.to_string(index=False))
    return iv_df


def run_feature_engineering(input_path: str = "data/cs-training.csv",
                             output_path: str = "data/processed.csv"):
    df = pd.read_csv(input_path, index_col=0)
    print(f"[加载] {len(df)} 行原始数据")
    df = clean_data(df)
    df = engineer_features(df)
    compute_all_iv(df)
    df.to_csv(output_path, index=False)
    print(f"\n[完成] 处理后数据已保存：{output_path}（{df.shape[0]} 行 × {df.shape[1]} 列）")
    return df
