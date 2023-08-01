import torch

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

EDGE_INDEX = torch.tensor([[ 0 , 1],
 [ 0 , 5],
 [ 0 , 9],
 [ 0 , 17],
 [ 1 , 2],
 [ 2 , 3],
 [ 3 , 4],
 [ 6 , 5],
 [ 6 , 7],
 [ 7 , 8],
 [ 9 ,10],
 [10 ,11],
 [11 ,12],
 [13 ,14],
 [14 ,15],
 [15 ,16],
 [17 ,18],
 [18 ,19],
 [19 ,20]])
N_NODE = 21
ADAPTIVE = True
ATTENTION = True

model = RecurrentGCN(3, 64, EDGE_INDEX, N_NODE, adaptive=ADAPTIVE, attention=ATTENTION)
model = model.to(device)

for i in model.parameters():
    print(i.is_cuda)