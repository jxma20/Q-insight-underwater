import json

import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from typing import List, Tuple
import pandas as pd
import os


def calculate_quality_metrics(
        ground_truth: List[float],
        predictions: List[float]
) -> Tuple[float, float, float, float]:
    """
    计算图像/视频质量评估中常用的四个指标：
    SRCC (Spearman's Rank Correlation Coefficient)
    KRCC (Kendall's Rank Correlation Coefficient)
    PLCC (Pearson's Linear Correlation Coefficient)
    RMSE (Root Mean Square Error)

    Args:
        ground_truth (List[float]): 真实的主观得分（例如 MOS/DMOS）。
        predictions (List[float]): 模型的预测得分。

    Returns:
        Tuple[float, float, float, float]: (SRCC, KRCC, PLCC, RMSE)
    """

    # 将列表转换为 NumPy 数组以便于计算
    y = np.array(ground_truth)
    y_hat = np.array(predictions)

    if len(y) != len(y_hat):
        raise ValueError("真实值和预测值的数量必须相等。")
    if len(y) < 2:
        # PLCC/SRCC/KRCC至少需要两个数据点
        raise ValueError("数据点数量必须至少为两个才能计算相关性。")

    ## 1. SRCC (Spearman's Rank Correlation Coefficient)
    # 衡量排名的相关性。scipy.stats.spearmanr 返回 (相关系数, p值)。
    # 我们只取相关系数。
    srcc, _ = spearmanr(y, y_hat)

    ## 2. KRCC (Kendall's Rank Correlation Coefficient)
    # 另一种衡量排名相关性的指标。scipy.stats.kendalltau 返回 (相关系数, p值)。
    # 我们只取相关系数。
    krcc, _ = kendalltau(y, y_hat)

    ## 3. PLCC (Pearson's Linear Correlation Coefficient)
    # 衡量线性关系。scipy.stats.pearsonr 返回 (相关系数, p值)。
    # 我们只取相关系数。
    plcc, _ = pearsonr(y, y_hat)

    ## 4. RMSE (Root Mean Square Error)
    # 衡量误差的均方根。
    # 计算公式：sqrt(sum((y - y_hat)^2) / N)
    rmse = np.sqrt(np.mean((y - y_hat) ** 2))

    return srcc, krcc, plcc, rmse


# --- 示例用法 ---
# 假设真实的主观得分 (MOS)
ground_truth_scores = pd.read_csv("/HOME/paratera_xy/pxy1092/HDD_POOL/Q-Insight/UWIQA/data.csv")["image_mos"]
# print(ground_truth_scores)

# Q_Insight iqa结果评估
with open("/HOME/paratera_xy/pxy1092/HDD_POOL/Q-Insight/src/eval/qinsightv3_uwiqa_result.json", 'r') as f:
    qinsight_res = json.load(f)
qinsight_res = [(int(k.split(".")[0]), v) for k, v in qinsight_res.items()]
predicted_scores = [i[1] for i in sorted(qinsight_res)]

try:
    srcc_val, krcc_val, plcc_val, rmse_val = calculate_quality_metrics(
        ground_truth_scores,
        predicted_scores
    )

    print("--- 质量评估指标结果 ---")
    # print(f"真实值 (y):     {ground_truth_scores}")
    # print(f"预测值 (y_hat): {predicted_scores}")
    print("-" * 30)
    print(f"SRCC (Spearman): {srcc_val:.4f}")
    print(f"KRCC (Kendall):  {krcc_val:.4f}")
    print(f"PLCC (Pearson):  {plcc_val:.4f}")
    print(f"RMSE (Error):    {rmse_val:.4f}")

except ValueError as e:
    print(f"计算错误: {e}")