
import torch


# target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
# output = torch.full([10, 64], 1.5)  # A prediction (logit)
# pos_weight = torch.ones([64])  # All weights are equal to 1
# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# print(output)
# print(target)

# print(output.shape)
# print(target.shape)
# loss = criterion(output, target)  # -log(sigmoid(1.5))
# print(loss)

#=======================================

from torch import nn

# loss = nn.MSELoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5)

# print(input)
# print(target)

# print(input.shape)
# print(target.shape)
# output = loss(input, target)
# output.backward()

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(input)
print(target)
print(input.shape)
print(target.shape)
output = loss(input, target)
output.backward()