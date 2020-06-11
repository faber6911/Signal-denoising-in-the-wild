import numpy as np
import librosa

def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def add_noise(clean_audio, noise_audio, snr):
    '''
    This function add a noise audio file to a clean audio one using
    a determined Speech Noise Ratio.
    
    @params:
    clean_audio (np.array): Clean audio file in np.array format
    noise_audio (np.array): Noise audio file in np.array format
    snr (float): Speech Noise Ratio in dB
    '''
    if len(clean_audio) >= len(noise_audio):
        while len(clean_audio) >= len(noise_audio):
            noise_audio = np.append(noise_audio, noise_audio)
    
    # Adjust noise length
    ind = np.random.randint(0, noise_audio.size - clean_audio.size)
    noiseSegment = noise_audio[ind:ind+clean_audio.size]
    
    # Signal Noise Ratio inspired by https://github.com/Sato-Kunihiko/audio-SNR
    RMSspeech = rms(clean_audio)
    RMSnoise = rms(noiseSegment)
    a = float(snr) / 20
    adjustedRMSnoise = RMSspeech / (10**a)
    
    # Noisy audio signal
    noisyAudio = clean_audio + noiseSegment*(adjustedRMSnoise/RMSnoise) 
    
    return noisyAudio
