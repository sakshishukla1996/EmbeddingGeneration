
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