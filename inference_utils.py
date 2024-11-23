from model import LightingModel
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import numpy as np

def find_activate_area(earthquake_history,earthquake_future,energy_predict,topk):
    energy_catalog_sum = earthquake_future.sum(axis = 1).cpu().numpy()
    topk_index = np.argsort(energy_catalog_sum)[-topk:]
    earthquake_history_topk = earthquake_history[:,topk_index,:].squeeze(-1).cpu().numpy()
    earthquake_future_topk = earthquake_future[topk_index].permute(1,0).cpu().numpy()
    energy_predict_topk = energy_predict[topk_index].permute(1,0).cpu().numpy()
    return earthquake_history_topk,earthquake_future_topk,energy_predict_topk,topk_index


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def calculate_earthquake_threshold(energy_predict_all,earthquake_future_all,threshold):
    earthquake_happen_predict = (energy_predict_all>threshold)
    earthquake_happen_real = (earthquake_future_all>threshold)
    return earthquake_happen_predict,earthquake_happen_real

def plot_confusion_matrix(energy_predict_all, earthquake_happen_all, threshold = 3.5):
    """
    计算并绘制混淆矩阵，同时打印各类评估指标。
    
    Args:
        energy_predict_all: 预测的能量值 (torch.Tensor)
        earthquake_happen_all: 地震发生的真实标签 (torch.Tensor)
    Returns:
        energy_predict: numpy 数组，预测的能量值
        earthquake_target: numpy 数组，地震发生的真实标签
    """
    labels = [0, 1]
    energy_predict = energy_predict_all.cpu().numpy().flatten()
    earthquake_target = earthquake_happen_all.cpu().numpy().flatten()

    # 将连续的预测值转为分类标签
    earthquake_happen_predict, earthquake_happen_real = calculate_earthquake_threshold(energy_predict, earthquake_target, threshold=threshold)

    # 计算混淆矩阵
    confusion_matrix_result = confusion_matrix(earthquake_happen_real, earthquake_happen_predict, labels=labels)

    # 提取混淆矩阵中的元素
    TP = confusion_matrix_result[1, 1]
    TN = confusion_matrix_result[0, 0]
    FP = confusion_matrix_result[0, 1]
    FN = confusion_matrix_result[1, 0]

    # 计算各类指标
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate
    TNR = TN / (FP + TN) if (FP + TN) > 0 else 0  # True Negative Rate
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    Error = (FP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    Aggregative_Score = (TPR + Precision + Accuracy) / 3

    # 绘制混淆矩阵
    sns.heatmap(confusion_matrix_result, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predict')
    plt.ylabel('Real')
    plt.title('Confusion Matrix')

    # 打印指标
    print(f"TPR (Recall): {TPR:.2f}")
    print(f"TNR (Specificity): {TNR:.2f}")
    print(f"Precision: {Precision:.2f}")
    print(f"Accuracy: {Accuracy:.2f}")
    print(f"Error: {Error:.2f}")
    print(f"FPR: {FPR:.2f}")
    print(f"Aggregative Score: {Aggregative_Score:.2f}")

    # 显示混淆矩阵
    plt.show()
    
    return energy_predict, earthquake_target



def plot_error(energy_predict,earthquake_target,error): 
    plt.figure(figsize=(10, 10))
    plt.scatter(earthquake_target,energy_predict, alpha=0.6, label="Data points")

    # 绘制基准线 y = x
    x = np.linspace(0, 7, 100)
    plt.plot(x, x, color="purple", linestyle="-", label="y=x")

    # 绘制 10% 误差线
    plt.plot(x, (1+error) * x, color="gray", linestyle="--", label=f"{error*100}% error bounds")
    plt.plot(x, (1-error) * x, color="gray", linestyle="--")

    # 填充误差范围
    plt.fill_between(x, (1-error) * x, (1+error) * x, color="gray", alpha=0.2)

    # 图表设置
    plt.xlabel("Real Log Energy", fontsize=14)
    plt.ylabel("Predicted Log Energy", fontsize=14)
    plt.title(f"Scatter Error", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlim(-0.2, 7)
    plt.ylim(-0.2, 7)

    # 显示图表
    plt.show()

def plot_delta_error(energy_predict,earthquake_target,error): 
    plt.figure(figsize=(10, 10))
    plt.scatter(earthquake_target,energy_predict, alpha=0.6, label="Data points")

    # 绘制基准线 y = x
    x = np.linspace(min(earthquake_target.min(),energy_predict.min()), max(earthquake_target.max(),energy_predict.max()), 100)
    plt.plot(x, x, color="purple", linestyle="-", label="y=x")

    # 绘制 10% 误差线
    plt.plot(x, (1+error) * x, color="gray", linestyle="--", label=f"{error*100}% error bounds")
    plt.plot(x, (1-error) * x, color="gray", linestyle="--")

    # 填充误差范围
    plt.fill_between(x, (1-error) * x, (1+error) * x, color="gray", alpha=0.2)

    # 图表设置
    plt.xlabel("Real delta_Log Energy", fontsize=14)
    plt.ylabel("Predicted delta_Log Energy", fontsize=14)
    plt.title(f"Scatter Delta Error", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # 显示图表
    plt.show()