from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def TSNE_plot(twodarray):
    #transform to 2D array
    X_embedded = TSNE(n_components=2, learning_rate='auto',init='random',random_state=0, perplexity=20).fit_transform(twodarray)  
    fig, ax = plt.subplots(figsize=(10,8))
    #Plotting 2D array
    ax.scatter(X_embedded[:,0], X_embedded[:,1], alpha=.5, color='blue')
    plt.title('Scatter plot using t-SNE')
    plt.show()