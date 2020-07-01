#!/root/anaconda3/envs/kalditorch/bin/python



#----- import packages

import argparse
import os
import sys
import pathlib
import re
import numpy as np
from sklearn.model_selection import train_test_split
import time


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='This program process the data from Siwis \
    database (https://www.unige.ch/lettres/linguistique/research/latl/siwis/database/) and Musan dataset \
    (https://www.openslr.org/17/) in order to develop an End-to-End system for signal denoising.')

    parser.add_argument('--main_path', type=str, required=True,
                        help='Absolute path to the folder that contain SIWIS and MUSAN datasets.')
    parser.add_argument('--lang', type=list, default=['EN'],
                        help='SIWIS contains different languages: English, Italian, French, Dutch. Default system use only English audios.')
    parser.add_argument('--min_length', type=float, required=True,
                        help='The minimum duration for the audio files to be considered from the system.')
    parser.add_argument('--test_size', type=float, default=.1,
                        help='Size for test set audios.')
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()
    
    #----- paths
    # SIWIS database https://www.unige.ch/lettres/linguistique/research/latl/siwis/database/
    # MUSAN dataset https://www.openslr.org/17/
    
    clean_audios = 'siwis_database/wav/'
    noise_audios = 'musan'
    info_clean_audios = 'siwis_database/info'
    
    clean_audios_path = os.path.join(args.main_path, clean_audios)
    noise_audios_path = os.path.join(args.main_path, noise_audios)
    print('\n\nClean audios (siwis_database) directory: {}'.format(clean_audios_path))
    print('Noise audios (musan dataset) directory: {}'.format(noise_audios_path))
    
    #----- utt2duration.scp file
    if args.verbose:
        print('\n\nChecking utt2duration.scp presence...')

    if not os.path.isdir('../data'):
        if args.verbose:
            print('Making data dir...')
        os.makedirs('../data')

    if not os.path.isfile('../data/utt2duration.scp'):
        if args.verbose:
            print('Making utt2duration.scp...')
        utt2duration = {}

        info_dir = os.listdir(os.path.join(args.main_path, info_clean_audios))
        audio_length_files = [elem for elem in info_dir if re.search('_audio_length.txt', elem)]

        with open('../data/utt2duration.scp', 'w') as outfile:
            durations = []
            for file in audio_length_files:
                for line in open(os.path.join(args.main_path, info_clean_audios, file)):
                    utt, duration = line.rstrip().split()
                    utt = utt.replace('.wav', '')
                    durations.append(float(duration))
                    outfile.write('{} {}\n'.format(utt, duration))
                    utt2duration[utt] = float(duration)

        outfile.close()
        if args.verbose:
            print('Clean audios measures:')
            print('Max: {}\nMin: {}\nMean: {}\nMedian: {}'.format(max(durations), min(durations),
                                                                  np.mean(durations), np.median(durations)))
    else:
        utt2duration = {}
        for line in open('../data/utt2duration.scp'):
            utt, duration = line.rstrip().split()
            utt2duration[utt] = float(duration)
        if args.verbose:
            print('/data/utt2duration.scp file already exists.')
        
        
        
    #----- train and test wav.scp
    utts, paths, spks = [], [], []
    
    for language in args.lang:
        for folder in os.listdir(os.path.join(clean_audios_path, language)):
            for utt in os.listdir(os.path.join(clean_audios_path, language, folder)):
                tmp = os.path.join(clean_audios_path, language, folder, utt)
                if pathlib.Path(tmp).suffix == '.wav':
                    if utt2duration[utt.replace('.wav', '')] > args.min_length:
                        utts.append(utt.replace('.wav', ''))
                        spks.append(utt.split('_')[2])
                        paths.append(tmp)

    X_train, X_test, y_train, y_test = train_test_split(np.array(paths), np.array(utts), test_size = args.test_size, stratify = spks)
    
    if args.verbose:
        print('\n\nChecking train and test wav.scp presence...')
    if not os.path.isdir('../data/train'):
        os.makedirs('../data/train')

    if not os.path.isfile('../data/train/wav.scp'):
        if args.verbose:
            print('Making train...')
        with open('../data/train/wav.scp', 'w') as outfile:
            for train_counter, line in enumerate(np.column_stack((y_train, X_train))):
                outfile.write('{} {}\n'.format(line[0], line[1]))

        outfile.close()

    else:
        if args.verbose:
            print('/data/train/wav.scp already exists.')

    if not os.path.isdir('../data/test'):
        os.makedirs('../data/test')

    if not os.path.isfile('../data/test/wav.scp'):
        if args.verbose:
            print('Making test...')
        with open('/data/test/wav.scp', 'w') as outfile:
            for test_counter, line in enumerate(np.column_stack((y_test, X_test))):
                outfile.write('{} {}\n'.format(line[0], line[1]))

        outfile.close()
        if args.verbose:
            print('Detected {} audio files'.format(train_counter+test_counter))
            print('{} in train'.format(train_counter))
            print('{} in test'.format(test_counter))

    else:
        if args.verbose:
            print('/data/test/wav.scp already exists.')
        
    #----- noise wav.scp
    if args.verbose:
        print('\n\nChecking for noise wav.scp presence...')

    if not os.path.isfile('../data/musan_noise.scp'):

        if args.verbose:
            print('Make musan.scp')
        counter = {}

        for folder in os.listdir(noise_audios_path):
            if os.path.isdir(os.path.join(noise_audios_path, folder)):
                if args.verbose:
                    print('Making musan_{}.scp'.format(folder))
                with open('../data/musan_{}.scp'.format(folder), 'w') as file:
                    for subfolder in os.listdir(os.path.join(noise_audios_path, folder)):
                        if os.path.isdir(os.path.join(noise_audios_path, folder, subfolder)):
                            for utt in os.listdir(os.path.join(noise_audios_path, folder, subfolder)):
                                if utt.endswith('.wav'):
                                    file.write('{} {}\n'.format(utt.replace('.wav', ''),
                                                                os.path.join(noise_audios_path, folder, subfolder, utt)))
                                    if folder in counter:
                                        counter[folder] += 1
                                    else:
                                        counter[folder] = 1

                file.close()
        if args.verbose:
            print('Detected {} noise audios'.format(sum(counter.values())))
            print(counter)

    else:
        if args.verbose:
            print('musan noise wav files already exists.')
    
    #----- utt2spk.scp
    if args.verbose:
        print('\n\nChecking for utt2spk.scp presence...')

    if not os.path.isfile('../data/utt2spk.scp'):

        list_of_speakers = set()
        with open('../data/utt2spk.scp', 'w') as file:
            for line in open('../data/train/wav.scp'):
                utt, path = line.split()
                spk = utt.split('_')[2]
                list_of_speakers.add(spk)
                file.write('{} {}\n'.format(utt, spk))

            for line in open('../data/test/wav.scp'):
                utt, path = line.split()
                spk = utt.split('_')[2]
                list_of_speakers.add(spk)
                file.write('{} {}\n'.format(utt, spk))

        file.close()
        if args.verbose:
            print('{} speakers'.format(len(list_of_speakers)))

    else:
        if args.verbose:
            print('/data/utt2spk.scp already exists.')
    end = time.time()
    
    print('{} executed in {:.2f} s'.format(sys.argv[0], (end-start)))