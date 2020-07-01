#!/root/anaconda3/envs/kalditorch/bin/python

#----- import packages
import argparse
import os
import kaldi_io
import numpy as np
import librosa
import random
from tqdm import tqdm
from utils import add_noise
import time
import sys

parser = argparse.ArgumentParser(description='This program process the audio files in the data folder, downsample, add noise creating\
                                              noisy audios and then split them using a window size chosen from the user. There is also the\
                                              possibility of using data augmentation using more than one noise, speech and music for\
                                              every clean audio file.')

parser.add_argument('--window_size', type=float, default = 2**14,
                        help='The size of the window used for slicing the audio files.')
parser.add_argument('--sample_rate', type=int, default=16000,
                    help='The sample rate used for the audio files.')
parser.add_argument('--stride', type=float, default=0.5,
                    help='Stride use during the audio splitting operation.')
parser.add_argument('--train_augmentation', type=int, default=1,
                    help='Augmentation for training data. Every audio file is augmented 3 x train_augmentation parameter.')
parser.add_argument('--test_augmentation', type=int, default=1, help='Augmentation for training data. Every audio file is augmented 3 x train_augmentation parameter.')
parser.add_argument('--verbose', default=False, action='store_true')

args = parser.parse_args()


#----- path to folders
clean_train_folder = '../data/train/'
clean_test_folder = '../data/test/'
serialized_train_folder = '../data/serialized_train_data/'
serialized_test_folder = '../data/serialized_test_data/'

#----- params
window_size = args.window_size  # about 1 second of samples
sample_rate = args.sample_rate
stride = args.stride
train_augmentation = args.train_augmentation
test_augmentation = args.test_augmentation
sample_rate = args.sample_rate


#----- def functions

def slice_signal(wav, window_size, stride):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 50%).
    """
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices

def process_and_serialize(data_type):
    """
    Serialize, down-sample, augment the sliced signals and save on separate folder.
    
    """
    noise_choice = {'music':659, 'noise':929, 'speech':425}
    
    if data_type == 'train':
        clean_folder = clean_train_folder
        serialized_folder = serialized_train_folder
        augmentation = train_augmentation
    else:
        clean_folder = clean_test_folder
        serialized_folder = serialized_test_folder
        augmentation = test_augmentation
        
    if not os.path.exists(serialized_folder):
        os.makedirs(serialized_folder)

    
    for line in tqdm(open(os.path.join(clean_folder, 'wav.scp')),
                     desc='Serialize and down-sample {} audios'.format(data_type)):
        
        utt, path = line.rstrip().split()
        clean_file, _ = librosa.load(path, sr = sample_rate)
        for noise_type in noise_choice:
            for aug in range(augmentation):
                noise_track = np.random.randint(0, noise_choice[noise_type])
                _, noise_path = open('../data/musan_{}.scp'.format(noise_type)).readlines()[noise_track].rstrip().split()
                noise_audio, _ = librosa.load(noise_path, sr = sample_rate)
                snr = random.choice([2.5, 7.5, 12.5, 17.5])
                noisy_file = add_noise(clean_file, noise_audio, snr=snr)

                # slice both clean signal and noisy signal
                clean_sliced = slice_signal(clean_file, window_size, stride)
                noisy_sliced = slice_signal(noisy_file, window_size, stride)
                # serialize - file format goes [original_file]_[noise_type]_[aug]_[slice_number].npy
                # ex) EN_C1_12_107.wav_0_5.npy denotes 5th slice of EN_C1_12_107.wav file in his augmentation version number 0
                for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                    pair = np.array([slice_tuple[0], slice_tuple[1]])
                    #print('{}.wav_{}_{}_{}'.format(utt, noise_type, aug, idx))
                    np.save(os.path.join(serialized_folder, '{}.wav_{}_{}_{}_{}'.format(utt, noise_type, snr, aug, idx)),
                            arr=pair)
                    
def data_verify(data_type):
    """
    Verifies the length of each data after pre-process.
    """
    if data_type == 'train':
        serialized_folder = serialized_train_folder
    else:
        serialized_folder = serialized_test_folder

    for root, dirs, files in os.walk(serialized_folder):
        for filename in tqdm(files, desc='Verify serialized {} audios'.format(data_type)):
            data_pair = np.load(os.path.join(root, filename))
            if data_pair.shape[1] != window_size:
                print('Snippet length not {} : {} instead'.format(window_size, data_pair.shape[1]))
                break
                
if __name__ == "__main__":
    
    start = time.time()
    
    flag = True
    try:
        directory = os.listdir(serialized_train_folder)
        if len(directory) > 0:
            flag = False
    except:
        pass
            
    if flag:        
        process_and_serialize('train')
        data_verify('train')
        process_and_serialize('test')
        data_verify('test')
    else:
        print('\n\nSerialized train and test data already exists')
    
    end = time.time()
    #print(flag)
    print('{} executed in {:.2f} s'.format(sys.argv[0], (end-start)))
    