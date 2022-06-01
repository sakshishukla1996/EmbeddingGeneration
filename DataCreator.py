from fileinput import filename
import librosa
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import matplotlib.pyplot as plt

class DataCreator(object):
    def __init__(self) -> None:
        pass
    
    def DirectoryCreator(self):
        return None

    
        
    def MelImageCreator(self, file_name):

        def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
            fig, axs = plt.subplots(1, 1)
            # axs.set_title(title or 'Spectrogram (db)')
            # axs.set_ylabel(ylabel)
            # axs.set_xlabel('frame')
            im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
            if xmax:
                axs.set_xlim((0, xmax))
            fig.colorbar(im, ax=axs)
            plt.show(block=False)
            fname = 'images/' + file_name+'.png'
            plt.savefig(fname)
            
        waveform, sample_rate = torchaudio.load(file_name)

        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 128

        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
        )

        melspec = mel_spectrogram(waveform)
        print(melspec)
        plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')
        return None

    def MetaDataMapper(self):
        return None
    