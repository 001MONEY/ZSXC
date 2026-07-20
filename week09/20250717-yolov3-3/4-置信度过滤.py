# 获取置信度 > 阈值的框的索引
# pred_mask = pred_re[:, :, :, :, 0] > threshold # threshold 通常取 0.5
# pred_idx = torch.where(pred_mask) # 返回符合条件位置的索引
# vectors = pred_re[pred_idx] # 只取这些位置的预测向量
# 案例：
import torch
a = torch.zeros((13,13,3,9)) #hw
a[5,6,0]=torch.tensor([0.8,0.2,0.8,0.5,0.6,0,0,0,1])
a[6,6,1] = torch.tensor([0.2,0.2,0.8,0.5,0.6,0,0,0,0.5])
tagert_mask = a[...,0]>0.4 #tagert_mask：torch.Size([13, 13, 3])
# print(a[tagert_mask])
pred_idx = torch.where(tagert_mask) #(tensor([5, 6]), tensor([6, 6]), tensor([0,
print(pred_idx)
print(a[pred_idx[0],pred_idx[1],pred_idx[2]])
# # 1]))
# vector = a[pred_idx]
# print(vector)
#tensor([[1.0000, 0.2000, 0.8000, 0.5000, 0.6000, 0.0000, 0.0000, 0.0000,
# 1.0000],
# # [1.0000, 0.2000, 0.8000, 0.5000, 0.6000, 0.0000, 0.0000, 0.0000,
# 0.5000]])