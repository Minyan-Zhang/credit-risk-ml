"""
main.py — 一键运行完整项目流水线
=====================================
运行方式：python main.py

流程：
  Step 1: EDA（探索性分析）
  Step 2: 数据清洗 + 特征工程
  Step 3: 模型训练与评估
  Step 4: 特征重要性分析
"""

import sys
import os

# 确保 src/ 可被导入
sys.path.insert(0, os.path.dirname(__file__))

from src.eda import load_data, basic_info, plot_target_distribution, \
    plot_feature_distributions, plot_correlation, plot_boxplot_by_target
from src.feature_engineering_module import run_feature_engineering
from src.modeling import run_modeling
from src.feature_importance import run_feature_importance


def main():
    print("\n" + "★" * 60)
    print("  基于机器学习的信贷违约风险评估系统")
    print("★" * 60)

    # ── Step 1: EDA ──────────────────────────────────────────
    print("\n\n【STEP 1】探索性数据分析 (EDA)")
    print("─" * 50)
    df_raw = load_data("data/cs-training.csv")
    basic_info(df_raw)
    plot_target_distribution(df_raw)
    plot_feature_distributions(df_raw)
    plot_correlation(df_raw)
    plot_boxplot_by_target(df_raw)

    # ── Step 2: 特征工程 ──────────────────────────────────────
    print("\n\n【STEP 2】数据清洗与特征工程")
    print("─" * 50)
    run_feature_engineering(
        input_path="data/cs-training.csv",
        output_path="data/processed.csv"
    )

    # ── Step 3: 建模 ──────────────────────────────────────────
    print("\n\n【STEP 3】模型训练与评估")
    print("─" * 50)
    results, X_test, y_test, features = run_modeling()

    # ── Step 4: 特征重要性 ────────────────────────────────────
    print("\n\n【STEP 4】特征重要性分析")
    print("─" * 50)
    run_feature_importance(features)

    # ── 最终总结 ──────────────────────────────────────────────
    print("\n\n" + "★" * 60)
    print("  项目运行完成！")
    print("★" * 60)
    print("\n  输出文件：")
    print("  ├── outputs/01_target_distribution.png    目标变量分布")
    print("  ├── outputs/02_feature_distributions.png  特征分布直方图")
    print("  ├── outputs/03_correlation_heatmap.png    相关性热力图")
    print("  ├── outputs/04_boxplot_by_target.png      分组箱线图")
    print("  ├── outputs/05_roc_curves.png             ROC 曲线对比")
    print("  ├── outputs/06_confusion_matrices.png     混淆矩阵")
    print("  ├── outputs/07_metrics_comparison.png     指标对比")
    print("  ├── outputs/08_xgboost_importance.png     XGBoost 特征重要性")
    print("  ├── outputs/09_rf_importance.png          RF 特征重要性")
    print("  ├── outputs/10_lr_coefficients.png        LR 系数")
    print("  ├── outputs/model_summary.csv             模型汇总表")
    print("  ├── models/LogisticRegression.pkl         LR 模型")
    print("  ├── models/RandomForest.pkl               RF 模型")
    print("  ├── models/XGBoost.pkl                    XGBoost 模型")
    print("  └── models/scaler.pkl                     标准化器\n")


if __name__ == "__main__":
    main()
