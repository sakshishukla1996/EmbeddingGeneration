## Speaker Embedding Generation using Denoising Diffusion Probalistic Models

First please install 'requirements.txt'

```bash
pip3 install -r requirements.txt
```

## Dataset

There are 3 types of embeddings generated from LibreSpeech Corpus: 
1. 64 Dimensional, which has 19k samples
2. 128 Dimensional, which has 49k samples
3. 704 Dimensional, which has 5k samples

## Training
```bash
python3 main.py
```

## Model

Linear and UNet Model are written in model.py file, which can be modified as per the requirement

UNet model Architecture
![UNet Architecture](https://github.tik.uni-stuttgart.de/FlorianLux/SpeakerEmbeddingGenerationDenoisingDiffusion/blob/master/figures/Unet.drawio.png)
## Output Audio Samples
These audio samples are generated after passing the generated embeddings to a TTS Engine. 


Female Voice:  

https://media.github.tik.uni-stuttgart.de/user/5258/files/afb624ab-1620-49a0-8119-2d4fa8310c27

Male Voice: 

https://media.github.tik.uni-stuttgart.de/user/5258/files/98ad29e5-c7ce-4e9d-a2c8-7badc3c942d5

## References






