from DataLoader import DataLoader
from DataCreator import DataCreator
from DataConverter import DataConverter
from GaussianSampler import GaussianSampler
from DiffusionModel import diffusionModel
from EmbeddingCreator import EmbeddingCreator
from Evaluation import Evaluation
import os



# Write Function Definitions, I/O , Functionalitie for all functions
#For Now Data is Loaded Manually
# data_download = DataLoader()
# data_path = data_download.DataDownloader()

#For Now Directory is Created Manually
data = DataCreator()
directories = data.DirectoryCreator()


for item in os.listdir('dataset'):
    filename = 'dataset/' + item
    data.MelImageCreator(filename)

# data.MetaDataMapper(directories)



