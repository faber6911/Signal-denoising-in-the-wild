import numpy as np
import torch
import librosa
import kaldi_io

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
    RMSspeech = np.sqrt(np.mean(clean_audio**2))
    RMSnoise = np.sqrt(np.mean(noiseSegment**2))
    a = float(snr) / 20
    adjustedRMSnoise = RMSspeech / (10**a)
    
    # Noisy audio signal
    noisyAudio = clean_audio + noiseSegment*(adjustedRMSnoise/RMSnoise) 
    
    return noisyAudio


class DenoiseDataset(torch.utils.data.Dataset):
    """PyTorch datalaoder for processing 'uncompressed' Kaldi feats.scp. It returns noisy audio
    with their corresponding clean audio file.
    """
    def __init__(self, scp_file, min_length):
        """
        Preprocess Kaldi feats.scp here
        @params:
        scp_file (path): Path to the scp file containing clean audio paths associated with
        corresponding noisy audios.
        min_length (float): Length in seconds for the desired audio files.
        """
        self.noisy_audios, self.clean_audios = [], []
        
        for line in open(scp_file):
            clean_audio_path, noisy_audio_path = line.rstrip().split() 
            self.noisy_audios.extend([noisy_audio_path])
            self.clean_audios.extend([clean_audio_path])
        
        self.noisy_audios = np.array(self.noisy_audios)
        self.clean_audios  = np.array(self.clean_audios)
        self.seq_len = min_length*16000
        print("Totally "+str(len(self.noisy_audios))+" samples")
    
    def __len__(self):
        """Return number of samples 
        """
        return len(self.clean_audios)

    def update(self, seq_len):
        """Update the self.seq_len. We call this in the main training loop 
        once per training iteration. 
        """
        self.seq_len = seq_len

    def __getitem__(self, index):
        """Generate samples
        """
        noisy_audio  = self.noisy_audios[index]
        full_mat = kaldi_io.read_mat(noisy_audio).squeeze(1)
        assert len(full_mat) >= self.seq_len
        pin = np.random.randint(0, len(full_mat) - self.seq_len + 1)
        chunk_mat = full_mat[pin:pin+self.seq_len]
        y, _ = librosa.load(self.clean_audios[index], sr = 16000)
        y = y[pin:pin+self.seq_len]
        
        return chunk_mat, y
