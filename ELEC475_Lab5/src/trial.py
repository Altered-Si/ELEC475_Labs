import torch
from torchvision import transforms


def euclidean_distance(predict, groud_truth):
    e_distance = torch.sqrt(torch.sum((predict - groud_truth) ** 2, dim=1))
    return e_distance


predicted = [[1, 2], [2, 3], [3, 4]]
ground_truth = [[2, 4], [2, 3], [5, 5]]
predicted = torch.Tensor(predicted)
ground_truth = torch.Tensor(ground_truth)

# print(predicted)
# print(ground_truth)
# print(predicted - ground_truth)

dist = euclidean_distance(predicted, ground_truth)
print('dist: ', dist)

total_distance = dist.sum().item()
print('total dist: ', total_distance)

loss_fn = torch.nn.L1Loss()
loss = loss_fn(predicted, ground_truth)
print('Loss: ', loss)

min_dist = (torch.min(dist)).item()
print('min dist: ', min_dist)
