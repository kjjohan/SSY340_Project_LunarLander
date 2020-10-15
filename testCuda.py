import torch as T
device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
print(device)