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



### SECTION 5

# 5.1

# compute log-likelihood for observation sequence on example data 
# (using pre-trained, concatenated, single-speaker 'o' HMM model)
obsloglik = log_multivariate_normal_density_diag(
    example['lmfcc'],
    isodigit_HMMs_onespkr['o']['means'],
    isodigit_HMMs_onespkr['o']['covars'],
)

# verification of consistent results
print('obsloglik shape:', obsloglik.shape, '| reference shape:', example['obsloglik'].shape)
print('Results are matching for observation log-likelihood:', np.allclose(obsloglik, example['obsloglik']))

# plotting for comparison of arbitrarily chosen digit
# Pick any utterance — let's use data[10], which the lab later tells us is "four"
utt = data[10]
digit = utt['digit']
print("Utterance digit:", utt['digit'])

obsloglik_utt = log_multivariate_normal_density_diag(
    utt['lmfcc'],
    isodigit_HMMs_onespkr[digit]['means'],
    isodigit_HMMs_onespkr[digit]['covars'],
)

plt.figure(figsize=(10, 4))
plt.pcolormesh(obsloglik_utt.T)
plt.title(f"Gaussian observation log-likelihood for utterance digit '{digit}'")
plt.xlabel("time frame")
plt.ylabel("state index")
plt.colorbar(label=r"$\log \phi_j(x_i)$")
plt.tight_layout()
plt.savefig('digit_obsloglik.png')



# 5.2

log_startprob = np.log(isodigit_HMMs_onespkr['o']['startprob'])
log_transmat = np.log(isodigit_HMMs_onespkr['o']['transmat'])
log_emlik = example['obsloglik']

log_alpha = forward(log_emlik, log_startprob, log_transmat)     # recursive forward algorithm
seq_log_lik = logsumexp(log_alpha[-1])      # observation seq likelihood is sum over all possible forward path probabilities

# verification of consistent results
print('Results are matching for forward algorithm:', np.allclose(log_alpha, example['logalpha']))
print('Results are matching for observation sequence likelihood:', np.allclose(seq_log_lik, example['loglik']))

# scoring of utterances with both pretrained world-level G-HMMs (single speaker vs entire dataset)
digits = list(isodigit_HMMs_onespkr.keys())

# -> single-speaker word-level HMMs
utt_log_liks_onespkr = np.empty(shape=(len(data), len(digits)))
for utt_idx, utt in enumerate(data):
    for digit_idx, digit in enumerate(digits):
        isodigit_hmm = isodigit_HMMs_onespkr[digit]
        obslik_utt_model = log_multivariate_normal_density_diag(
            utt['lmfcc'],
            isodigit_hmm['means'],
            isodigit_hmm['covars'],
        )
        logalpha_utt_model = forward(
            obslik_utt_model,
            np.log(isodigit_hmm['startprob']),
            np.log(isodigit_hmm['transmat'])
        )

        utt_log_liks_onespkr[utt_idx, digit_idx] = logsumexp(logalpha_utt_model[-1])

predicted_digits_onespkr = [digits[max_idx] for max_idx in np.argmax(utt_log_liks_onespkr, axis=1)]

print('Predicted digits (single speaker models)', predicted_digits_onespkr)     # quite bad for male speaker (obviously)


# -> all-speakers word-level HMMs
utt_log_liks_all = np.empty(shape=(len(data), len(digits)))
for utt_idx, utt in enumerate(data):
    for digit_idx, digit in enumerate(digits):
        isodigit_hmm = isodigit_HMMs_all[digit]
        obslik_utt_model = log_multivariate_normal_density_diag(
            utt['lmfcc'],
            isodigit_hmm['means'],
            isodigit_hmm['covars'],
        )
        logalpha_utt_model = forward(
            obslik_utt_model,
            np.log(isodigit_hmm['startprob']),
            np.log(isodigit_hmm['transmat'])
        )

        utt_log_liks_all[utt_idx, digit_idx] = logsumexp(logalpha_utt_model[-1])

predicted_digits_all = [digits[max_idx] for max_idx in np.argmax(utt_log_liks_all, axis=1)]
true_digit_seq = [utt['digit'] for utt in data]

print('Predicted digits (all speakers models)', predicted_digits_all)       # quite on point!
print('True digit sequence in dataset:', true_digit_seq)                    # first man, then female speaker



# 5.3