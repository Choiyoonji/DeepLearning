import torch

# 텐서 연산 테스트
a = torch.tensor([[1, 2], [3, 4]]).to('cuda')
b = torch.tensor([[5, 6], [7, 8]]).to('cuda')
c = a + b
print(c)
