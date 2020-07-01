import numpy as np
import librosa
import os

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

class ScheduledOptim(object):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        "Learning rate scheduling per step"

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def state_dict(self):
        ret = {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
        }
        ret['optimizer'] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])
        
        
def change_path_scp(infile, outfile, old_abs_path, new_abs_path, separator):
  '''
  Change the abs path in the SCP file in order to point to the associated ark file.
  @Params:
    infile (path): path to the train.scp file.
    outfile (path): path to the train<Machine>.scp file.
    old_abs_path (path): the old abs path. ['/home/faber6911/kaldi/egs/'].
    new_abs_path (path): depends to the path where the repository is cloned.
  '''
  with open(outfile, 'w') as f:
    for line in open(infile):
      utt, path = line.rstrip().split()
      path = path.replace(old_abs_path, new_abs_path)
      #print(utt, path)
      f.write('{}{}{}\n'.format(utt, separator, path))

  f.close()

  print('Done')

  print(os.popen('head {}'.format(outfile)).read())