"""
03_modeling.py — 模型训练与评估
=====================================
运行方式：python src/03_modeling.py
输出：
  - models/ 目录下保存的模型文件
  - outputs/ 目录下的评估图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score,
    f1_score, classification_report, roc_curve, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.feature_engineering import get_feature_columns   # 复用特征列定义

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42


# ──────────────────────────────────────────
# 1. 数据准备
# ──────────────────────────────────────────

def prepare_data(path: str = "data/processed.csv"):
    """加载处理后的数据，划分训练/测试集"""
    df = pd.read_csv(path)
    features = get_feature_columns()

    # 过滤掉实际不存在的列
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"[数据划分] 训练集：{X_train.shape[0]} 行，测试集：{X_test.shape[0]} 行")
    print(f"[数据划分] 训练集违约率：{y_train.mean():.2%}，测试集违约率：{y_test.mean():.2%}")

    # SMOTE 过采样（仅对训练集）
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE] 过采样后训练集：{X_train_res.shape[0]} 行，违约率：{y_train_res.mean():.2%}")

    return X_train, X_test, y_train, y_test, X_train_res, y_train_res, features


# ──────────────────────────────────────────
# 2. 模型定义
# ──────────────────────────────────────────

def get_models():
    """返回三个模型"""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=14,      # 正负样本比约 14:1
            eval_metric="auc",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            verbosity=0
        ),
    }
    return models


# ──────────────────────────────────────────
# 3. 训练与评估
# ──────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test, X_train_res, y_train_res):
    """训练所有模型并汇总评估指标"""
    models = get_models()
    results = {}

    # Logistic Regression 需要标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    for name, model in models.items():
        print(f"\n{'='*55}")
        print(f"  训练模型：{name}")
        print(f"{'='*55}")

        if name == "LogisticRegression":
            model.fit(X_train_scaled, y_train_res)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_res, y_train_res)
            y_prob = model.predict_proba(X_test)[:, 1]

        # 使用最优阈值（最大化 F1 或业务需要可调整）
        threshold = find_best_threshold(y_test, y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "threshold": threshold,
            "AUC": round(auc, 4),
            "Recall": round(recall, 4),
            "Precision": round(precision, 4),
            "F1": round(f1, 4),
        }

        print(f"  AUC       : {auc:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  最优阈值  : {threshold:.2f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['正常(0)', '违约(1)'])}")

        # 保存模型
        joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")
        print(f"  [保存] models/{name}.pkl")

    return results


def find_best_threshold(y_true, y_prob):
    """寻找最优分类阈值（最大化 F1）"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh


# ──────────────────────────────────────────
# 4. 可视化
# ──────────────────────────────────────────

def plot_roc_curves(results: dict, y_test):
    """绘制所有模型的 ROC 曲线"""
    plt.figure(figsize=(8, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['AUC']:.4f})", color=color, lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("ROC Curve Comparison", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_roc_curves.png", dpi=150)
    plt.close()
    print(f"[图表] 保存：{OUTPUT_DIR}/05_roc_curves.png")


def plot_confusion_matrices(results: dict, y_test):
    """绘制混淆矩阵"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Default"])
        ax.set_yticklabels(["Normal", "Default"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax)

    plt.suptitle("Confusion Matrices", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_confusion_matrices.png", dpi=150)
    plt.close()
    print(f"[图表] 保存：{OUTPUT_DIR}/06_confusion_matrices.png")


def plot_metrics_comparison(results: dict):
    """各模型指标对比柱状图"""
    metrics = ["AUC", "Recall", "Precision", "F1"]
    names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_metrics_comparison.png", dpi=150)
    plt.close()
    print(f"[图表] 保存：{OUTPUT_DIR}/07_metrics_comparison.png")


def print_summary_table(results: dict):
    """打印最终对比表"""
    print("\n" + "=" * 60)
    print("  模型性能汇总")
    print("=" * 60)
    rows = []
    for name, res in results.items():
        rows.append({
            "模型": name,
            "AUC": res["AUC"],
            "Recall": res["Recall"],
            "Precision": res["Precision"],
            "F1": res["F1"],
            "阈值": round(res["threshold"], 2),
        })
    summary = pd.DataFrame(rows).set_index("模型")
    print(summary.to_string())
    summary.to_csv(f"{OUTPUT_DIR}/model_summary.csv")
    print(f"\n[完成] 汇总表已保存：{OUTPUT_DIR}/model_summary.csv")


# ──────────────────────────────────────────
# 5. 主流程
# ──────────────────────────────────────────

def run_modeling():
    X_train, X_test, y_train, y_test, X_train_res, y_train_res, features = prepare_data()
    results = train_and_evaluate(X_train, X_test, y_train, y_test, X_train_res, y_train_res)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_metrics_comparison(results)
    print_summary_table(results)
    return results, X_test, y_test, features


if __name__ == "__main__":
    run_modeling()
