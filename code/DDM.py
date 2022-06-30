import wandb

wandb.init(project="DDM-Project")
WANDB_API_KEY = 'ffe5b918b921d391434d044c9bc030bdef3d48de'
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from TSNE_Plot import TSNE_plot
from dataNormalize import dataNormalize

from model import ConditionalModel
from ema import EMA
import torch.optim as optim

from utils import * 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Acquired GPU successfully")
# device = torch.device("cuda:2,3,4" if torch.cuda.is_available() else "cpu")
DATA_PATH = '../dataset/embedding_vectors_as_list.pt'
list_of_embeddings =torch.load(DATA_PATH, map_location='cpu')

#Taking first 100 items from speaker embedding list and running the experiment
ten_emb = list_of_embeddings[:100]
b = torch.stack(ten_emb)
c = b.detach().numpy()
c = dataNormalize(c)

TSNE_plot(c, color='red', name='BeforeDDM.png')    

print("The shape of input is: ", c.shape)

torch.set_default_dtype(torch.float64)
dataset = torch.tensor(c)

num_steps = 100
batch_size = 64
betas = torch.tensor([1.7e-5] * num_steps)
betas = make_beta_schedule(schedule='sigmoid', n_timesteps=num_steps, start=1e-5, end=0.5e-2)

alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

def q_x(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)

posterior_mean_coef_1 = (betas * torch.sqrt(alphas_prod_p) / (1 - alphas_prod))
posterior_mean_coef_2 = ((1 - alphas_prod_p) * torch.sqrt(alphas) / (1 - alphas_prod))
posterior_variance = betas * (1 - alphas_prod_p) / (1 - alphas_prod)
posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].view(1, 1), posterior_variance[1:].view(-1, 1)), 0)).view(-1)

def q_posterior_mean_variance(x_0, x_t, t):
    coef_1 = extract(posterior_mean_coef_1, t, x_0)
    coef_2 = extract(posterior_mean_coef_2, t, x_0)
    mean = coef_1 * x_0 + coef_2 * x_t
    var = extract(posterior_log_variance_clipped, t, x_0)
    return mean, var



model = ConditionalModel(num_steps)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#dataset = torch.tensor(data.T).float()
# Create EMA model
ema = EMA(0.9)
ema.register(model)

# Batch size
wandb.config = {
  "epochs": num_steps,
  "batch_size": batch_size
}

for t in range(1000):
    # X is a torch Variable
    permutation = torch.randperm(dataset.size()[0])
    for i in range(0, dataset.size()[0], batch_size):
        # Retrieve current batch
        indices = permutation[i:i+batch_size]
        batch_x = dataset[indices]
        # Compute the loss.
        loss = noise_estimation_loss(model, batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
        wandb.log({"loss": loss})
        # Optional
        wandb.watch(model)
        # Before the backward pass, zero all of the network gradients
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        # Calling the step function to update the parameters
        optimizer.step()
        # Update the exponential moving average
        ema.update(model)
    # Print loss
    if (t % 100 == 0):
        print(loss)
        x_seq = p_sample_loop(model, dataset.shape,num_steps,alphas,betas,one_minus_alphas_bar_sqrt)
        fig, axs = plt.subplots(1, 10, figsize=(28, 3))
        for i in range(1, 11):
            cur_x = x_seq[i * 10].detach()
            axs[i-1].scatter(cur_x[:, 0], cur_x[:, 1],color='white',edgecolor='gray', s=5);
            axs[i-1].set_axis_off(); 
            axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*100)+'})$')


print(len(x_seq))
d = cur_x.detach().numpy()
d = dataNormalize(d)
TSNE_plot(d, color='blue', name='AfterDDM.png')

#Now Plotting both on same graph with different colors
#Plotting 2D array
c = TSNE(n_components=2, learning_rate='auto',init='random',random_state=0, perplexity=20).fit_transform(c)  
d = TSNE(n_components=2, learning_rate='auto',init='random',random_state=0, perplexity=20).fit_transform(d)  
wandb.log({'Original Data: ':c})
wandb.log({'Generated Data: ':d})
fig, ax = plt.subplots(figsize=(10,8))
#Plotting 2D array
ax.scatter(c[:,0], c[:,1], alpha=.5, color='red')
ax.scatter(d[:,0], d[:,1], alpha=.5, color='blue')
plt.title('Scatter plot using t-SNE')
# plt.show()
wandb.log({"chart": plt})


plt.savefig('AfterDDMBoth.png')