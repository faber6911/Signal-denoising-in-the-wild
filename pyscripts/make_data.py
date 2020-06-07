#!/root/anaconda3/envs/kalditorch/bin/python



#----- import packages

import argparse
import os
import sys
import pathlib
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program process the data from Siwis \
    database (https://www.unige.ch/lettres/linguistique/research/latl/siwis/database/) and Musan dataset \
    (https://www.openslr.org/17/) in order to develop an End-to-End system for signal denoising.')

    parser.add_argument('--main_path', type=str, help='Absolute path to the folder that contain SIWIS and MUSAN datasets.')
    parser.add_argument('--lang', type=str, default='EN', help='SIWIS contains different languages: English, Italian, French, Dutch.\
                        Default system use only English audios.')

    args = parser.parse_args()
    print(args.main_path, args.lang)