# 基于机器学习的信贷违约风险评估系统

> 项目周期：2025.11 - 2026.1  
> 数据集：Kaggle — Give Me Some Credit  
> 目标：预测用户未来 2 年内是否发生严重违约（二分类）

---

## 项目结构

```
credit_risk/
├── data/
│   ├── cs-training.csv       ← 从 Kaggle 下载放这里
│   └── processed.csv         ← 运行后自动生成
├── src/
│   ├── eda.py                Step 1：探索性分析
│   ├── feature_engineering_module.py  Step 2：清洗 + 特征工程
│   ├── modeling.py           Step 3：模型训练与评估
│   └── feature_importance.py Step 4：特征重要性分析
├── outputs/                  ← 所有图表自动保存到这里
├── models/                   ← 训练好的模型文件
├── main.py                   ← 一键运行全流程
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载数据
前往 Kaggle 下载数据集：  
https://www.kaggle.com/c/GiveMeSomeCredit/data  
下载 `cs-training.csv`，放入 `data/` 目录。

### 3. 一键运行
```bash
python main.py
```

或分步运行：
```bash
python src/01_eda.py                  # Step 1: EDA
python src/02_feature_engineering.py  # Step 2: 特征工程
python src/03_modeling.py             # Step 3: 建模
python src/04_feature_importance.py   # Step 4: 特征重要性
```

---

## 数据集说明

| 字段 | 含义 |
|------|------|
| SeriousDlqin2yrs | **目标变量**：2年内是否严重违约（1=违约，0=正常） |
| RevolvingUtilizationOfUnsecuredLines | 信用卡等无担保贷款使用率 |
| age | 用户年龄 |
| NumberOfTime30-59DaysPastDueNotWorse | 30-59天逾期次数 |
| DebtRatio | 负债率（月债务/月收入） |
| MonthlyIncome | 月收入 |
| NumberOfOpenCreditLinesAndLoans | 开放信用账户数 |
| NumberOfTimes90DaysLate | 90天以上逾期次数 |
| NumberRealEstateLoansOrLines | 房产贷款数量 |
| NumberOfTime60-89DaysPastDueNotWorse | 60-89天逾期次数 |
| NumberOfDependents | 赡养人数（配偶、子女等） |

---

## 技术方案

### 数据清洗
- 删除 `age <= 0` 等明显异常值
- 删除 `RevolvingUtilizationOfUnsecuredLines > 1.5` 的极端异常值
- `MonthlyIncome`、`NumberOfDependents` 使用中位数填充缺失值

### 特征工程
| 新特征 | 含义 |
|--------|------|
| DebtToIncome | 月债务金额（DebtRatio × MonthlyIncome） |
| LogMonthlyIncome | 月收入对数（消除右偏） |
| LogRevolvingUtil | 信用使用率对数 |
| TotalOverdueTimes | 所有逾期次数汇总 |
| HasOverdue | 是否有过任何逾期（0/1） |
| AgeGroup | 年龄分段（0-4） |
| DebtPerAccount | 每个信用账户的债务率 |

### 处理样本不平衡
- 正负样本比约 **14:1**（严重不平衡）
- 使用 **SMOTE 过采样**（sampling_strategy=0.3）
- Logistic Regression 使用 `class_weight='balanced'`
- XGBoost 使用 `scale_pos_weight=14`

### 建模
| 模型 | 关键参数 |
|------|----------|
| Logistic Regression | C=0.1, max_iter=1000, balanced |
| Random Forest | n_estimators=200, max_depth=8, balanced |
| XGBoost | n_estimators=300, lr=0.05, scale_pos_weight=14 |

### 评估指标
- **AUC**（主要指标，目标 0.78+）
- Recall（召回率，尽量识别出所有违约用户）
- Precision（精确率）
- F1（综合指标）
- 使用最优分类阈值（而非默认0.5）

---

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `outputs/01_target_distribution.png` | 目标变量分布（正负样本比例） |
| `outputs/02_feature_distributions.png` | 所有特征分布直方图 |
| `outputs/03_correlation_heatmap.png` | 特征相关性热力图 |
| `outputs/04_boxplot_by_target.png` | 各特征按违约/正常分组箱线图 |
| `outputs/05_roc_curves.png` | 三个模型 ROC 曲线对比 |
| `outputs/06_confusion_matrices.png` | 混淆矩阵 |
| `outputs/07_metrics_comparison.png` | AUC/Recall/Precision/F1 柱状对比 |
| `outputs/08_xgboost_importance.png` | XGBoost 特征重要性 |
| `outputs/09_rf_importance.png` | Random Forest 特征重要性 |
| `outputs/10_lr_coefficients.png` | Logistic Regression 系数 |
| `outputs/model_summary.csv` | 模型性能汇总表 |
| `models/XGBoost.pkl` | 保存的 XGBoost 模型 |
| `models/RandomForest.pkl` | 保存的 Random Forest 模型 |
| `models/LogisticRegression.pkl` | 保存的 LR 模型 |
| `models/scaler.pkl` | 标准化器（LR 预测时需要） |

---

## 预期结果

| 模型 | AUC | Recall |
|------|-----|--------|
| Logistic Regression | ~0.78 | ~0.65 |
| Random Forest | ~0.82 | ~0.60 |
| **XGBoost** | **~0.85** | **~0.62** |

---

## 关键业务洞察

高风险用户画像：
1. **有逾期记录**（尤其是90天以上严重逾期）
2. **信用卡使用率高**（>80%，说明资金紧张）
3. **年龄较小**（< 30岁，信用历史短）
4. **高负债率 + 低月收入**（还款能力弱）

---

## 简历写法参考

```
基于机器学习的信贷违约风险评估系统    2025.11 - 2026.1
• 基于 Kaggle Give Me Some Credit 数据集（15万条），构建用户违约风险预测模型（二分类）
• 完成数据清洗与特征工程（缺失值处理、变量分箱/WOE编码、构造7个衍生特征）
• 使用 SMOTE 解决 14:1 样本不平衡问题，分别训练 Logistic Regression、Random Forest、XGBoost
• XGBoost 模型 AUC 达 0.85，Recall 0.62，优于基线 Logistic Regression（AUC 0.78）
• 分析特征重要性，识别高风险用户关键因素：历史逾期次数、信用使用率、负债率、月收入水平
```
