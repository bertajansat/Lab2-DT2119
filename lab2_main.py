### import of variables or functions from other lab files
from prondict import prondict
from lab2_proto import *
from lab2_tools import *

import numpy as np
import matplotlib.pyplot as plt



### SECTION 4

# load utterance subdataset from TIDIGITS 
# -> contains liftered MFCC features amongst others
data = np.load('lab2_data.npz', allow_pickle=True)['data']
example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()      # for double-checking

# load pre-trained HMM model sets (single speaker vs entire TIDIGITS dataset)
# -> contains 21 phoneme models, each being a phoneme-level Gaussian HMM with diagonal covariance
phoneHMMs_onespkr = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
phoneHMMs_all = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()

# create world-level HMMs (left-to-right concat of phoneme-level HMMs)
# -> isolated spoken digits should also include silence at start and end of utterance
isodigits2phonemes = { digit: ['sil'] + prondict[digit] + ['sil'] for digit in prondict}

isodigit_HMMs_onespkr = { isodigit: concatHMMs(phoneHMMs_onespkr, isodigits2phonemes[isodigit]) for isodigit in isodigits2phonemes}
isodigit_HMMs_all = { isodigit: concatHMMs(phoneHMMs_all, isodigits2phonemes[isodigit]) for isodigit in isodigits2phonemes}