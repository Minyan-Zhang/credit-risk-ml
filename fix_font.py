"""
fix_font.py — 修复 app.py 中 matplotlib 中文乱码问题
把图表里的中文全换成英文，Streamlit 页面文字不受影响
运行方式：python fix_font.py
"""
import re

with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. 修复字体配置：强制使用支持中文的字体（如果有），否则回退英文
old_font = '''matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False'''

new_font = '''import matplotlib.font_manager as fm
# 尝试找系统中文字体，找不到就用英文标签
_zh_fonts = [f.name for f in fm.fontManager.ttflist
             if any(kw in f.name for kw in ["SimHei","Microsoft YaHei","PingFang","Heiti","WenQuanYi","Noto"])]
if _zh_fonts:
    matplotlib.rcParams["font.sans-serif"] = [_zh_fonts[0]] + ["DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False'''

code = code.replace(old_font, new_font)

# 2. 把 FEAT_LABELS 全换成英文
old_labels = '''FEAT_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "信用卡使用率",
    "age":                                  "年龄",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59天逾期次数",
    "DebtRatio":                            "负债率",
    "MonthlyIncome":                        "月收入",
    "NumberOfOpenCreditLinesAndLoans":      "开放信用账户数",
    "NumberOfTimes90DaysLate":              "90天+逾期次数",
    "NumberRealEstateLoansOrLines":         "房产贷款数",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89天逾期次数",
    "NumberOfDependents":                   "赡养人数",
    "DebtToIncome":                         "债务收入比",
    "LogMonthlyIncome":                     "月收入(对数)",
    "LogRevolvingUtil":                     "信用使用率(对数)",
    "TotalOverdueTimes":                    "逾期总次数",
    "HasOverdue":                           "是否有逾期",
    "AgeGroup":                             "年龄段",
    "DebtPerAccount":                       "每账户债务率",
}'''

new_labels = '''FEAT_LABELS = {
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
}'''

code = code.replace(old_labels, new_labels)

# 3. 修复图表里的中文标题和标签
replacements = {
    # plot_target_dist
    '"正常(0)"': '"Normal (0)"',
    '"违约(1)"': '"Default (1)"',
    '"样本数量"': '"Count"',
    '"样本比例"': '"Proportion"',
    '"目标变量分布"': '"Target Variable Distribution"',
    # plot_feat_dist
    '"原始特征分布（截断99%分位）"': '"Feature Distributions (clipped at 99th pct)"',
    # plot_corr
    '"特征相关性热力图"': '"Feature Correlation Heatmap"',
    # plot_roc
    '"假正率 (FPR)"': '"False Positive Rate (FPR)"',
    '"真正率 (TPR / Recall)"': '"True Positive Rate (Recall)"',
    '"ROC 曲线对比"': '"ROC Curve Comparison"',
    '"随机  AUC=0.50"': '"Random  AUC=0.50"',
    # plot_cm
    '"预测值"': '"Predicted"',
    '"真实值"': '"Actual"',
    '"正常"': '"Normal"',
    '"违约"': '"Default"',
    '"混淆矩阵"': '"Confusion Matrices"',
    # plot_metrics
    '"得分"': '"Score"',
    '"模型性能对比"': '"Model Performance Comparison"',
    # plot_importance
    '"特征重要性"': '"Feature Importance"',
    # plot_lr_coef
    '"系数（正值 → 违约风险升高）"': '"Coefficient (positive = higher default risk)"',
    '"Logistic Regression 系数"': '"Logistic Regression Coefficients"',
    '"风险升高"': '"Risk Increase"',
    '"风险降低"': '"Risk Decrease"',
    # plot_boxplot
    '"正常(0)"': '"Normal (0)"',
    '"违约(1)"': '"Default (1)"',
    # predict bar
    '"违约概率"': '"Default Probability"',
    '"违约概率可视化"': '"Default Probability Visualization"',
}

for zh, en in replacements.items():
    code = code.replace(zh, en)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 修复完成！现在重新运行：streamlit run app.py")
