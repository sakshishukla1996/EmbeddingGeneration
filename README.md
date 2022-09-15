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

https://media.github.tik.uni-stuttgart.de/user/5258/files/dd0d0b7e-e4fc-4327-99ca-29f1bb1c3a85


Male Voice: 

https://media.github.tik.uni-stuttgart.de/user/5258/files/06631b39-0a5e-4342-89fd-0877007072a9





