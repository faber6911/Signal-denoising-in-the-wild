#!/root/anaconda3/envs/kalditorch/bin/python

#----- import packages

import argparse
import os
import kaldi_io
import numpy as np
import librosa
import IPython.display as ipd
import random
import kaldiio
from kaldiio import WriteHelper
from utils import add_noise
import time
import sys


parser = argparse.ArgumentParser(description='This program preprocess audio data and create two ark files, one for train and one for test data, with associated scp files. The clean audio and the noisy audio are associated with the label of the speaker.')

parser.add_argument('--train_augmentation', type=int, default=1,
                    help='The augmentation parameter for the training set. The parameter is multiplied by 3.')
parser.add_argument('--test_augmentation', type=int, default=1,
                    help='The augmentation parameter for the test set. The parameter is multiplied by 3.')
parser.add_argument('--compression_method', type=int, default=3,
                    help='The compression method used by kaldiio for the ark files. Default is 3, available values are [1, 2, 3].')
parser.add_argument('--sample_rate', type=int, default=16000,
                    help='The sample rate used for the import from librosa.')
parser.add_argument('--verbose', default=False, action='store_true')

args = parser.parse_args()


#----- paths
ark_path_train = '../data/train/train.ark'
scp_path_train = '../data/train/train.scp'
ark_path_test = '../data/test/test.ark'
scp_path_test = '../data/test/test.scp'

#----- parameters
train_augmentation = args.train_augmentation
test_augmentation = args.test_augmentation
compression_method = args.compression_method
sample_rate = args.sample_rate


#----- corpus
if __name___ == "__main__":
    
    start = time.time()
    
    if not os.path.isfile('../data/train/train.ark'):
    
        print('\n\nAdding noise to train audio and augmenting each file {} times'.format(train_augmentation*3))
        writer = WriteHelper('ark,scp:{},{}'.format(ark_path_train, scp_path_train), compression_method=compression_method)

        noise_choice = {'music':659, 'noise':929, 'speech':425}

        for count, line in enumerate(open('../data/train/wav.scp')):
            # clean audio path
            utt, path = line.rstrip().split()
            # clean audio file
            clean_audio, _ = librosa.load(path, sr = sample_rate)
            # now for every noise type we augment n times the clean audio file using random noise audio files
            for noise_type in noise_choice:
                for aug in range(train_augmentation):
                    noise_track = np.random.randint(0, noise_choice[noise_type])
                    _, noise_path = open('../data/musan_{}.scp'.format(noise_type)).readlines()[noise_track].rstrip().split()
                    noise_audio, _ = librosa.load(noise_path, sr = sample_rate)
                    noisy_audio = add_noise(clean_audio, noise_audio, snr=random.choice([2.5, 7.5, 12.5, 17.5]))
                    # write ark and associated scp file in train directory
                    writer(utt, np.concatenate((clean_audio.reshape(1, -1), noisy_audio.reshape(1, -1))))
            if count % 100 == 0:
                if args.verbose:
                    print('Augmented {} audio files'.format(count))

        writer.close()
        print('\nAugmented train data')

        print('\n\nAdding noise to test audio and augmenting each file {} times'.format(test_augmentation*3))
        writer = WriteHelper('ark,scp:{},{}'.format(ark_path_test, scp_path_test), compression_method=compression_method)

        noise_choice = {'music':659, 'noise':929, 'speech':425}

        for count, line in enumerate(open('../data/test/wav.scp')):
            # clean audio path
            utt, path = line.rstrip().split()
            # clean audio file
            clean_audio, _ = librosa.load(path, sr = sample_rate)
            # now for every noise type we augment n times the clean audio file using random noise audio files
            for noise_type in noise_choice:
                for aug in range(test_augmentation):
                    noise_track = np.random.randint(0, noise_choice[noise_type])
                    _, noise_path = open('../data/musan_{}.scp'.format(noise_type)).readlines()[noise_track].rstrip().split()
                    noise_audio, _ = librosa.load(noise_path, sr = sample_rate)
                    noisy_audio = add_noise(clean_audio, noise_audio, snr=random.choice([2.5, 7.5, 12.5, 17.5]))
                    # write ark and associated scp file in train directory
                    writer(utt, np.concatenate((clean_audio.reshape(1, -1), noisy_audio.reshape(1, -1))))
            if count % 100 == 0:
                if args.verbose:
                    print('Augmented {} audio files'.format(count))

        writer.close()
        print('\nAugmented test data')
    else:
        print('ark and scp files already exists in /data/train and /data/test folders.')

    end = time.time()
    
    print('{} executed in {:.2f} s'.format(sys.argv[0], (end-start)))