
import torch
import numpy as np
from TSNE_Plot import TSNE_plot
from dataNormalize import dataNormalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA_PATH = '../dataset/704dim_embeds.pt'
DATA_PATH = '../emebddings_after_noise_sampling/704_1st_test.pt'
list_of_embeddings =torch.load(DATA_PATH, map_location='cpu')
print(list_of_embeddings[0][0])
print(list_of_embeddings[0][0].shape)

sample = list_of_embeddings[0][0]
print(len(sample))
sample[0] = 0
sample[1] = 0
sample[2] = 0
sample[3] = 0
sample[4] = 0
print(sample)
# sample = list_of_embeddings[0][0]
# sample = torch.stack(sample)
# sample = sample.detach().numpy()
# w, v = np.linalg.eig(sample)
# print(len(v))
#v is eigen vector
print(torch.ones(704))

# r5 = torch.randint(low=0.5, high=1, size=(704))
# print(r5)
# x = torch.empty((704), dtype=torch.float)
# print(x)
y = torch.full((1,704), 0.2, dtype=torch.float64)
print(y)

