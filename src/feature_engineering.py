"""
feature_engineering.py — 供其他模块导入的公共函数
"""


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
