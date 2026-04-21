import numpy as np
from lab2_tools import *


# already implemented
def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output:
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
            means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    num_states_hmm1 = len(hmm1['startprob']) - 1
    num_states_hmm2 = len(hmm2['startprob']) - 1
    num_states_concat = num_states_hmm1 + num_states_hmm2 + 1
    
    startprob = np.concatenate((hmm1['startprob'], hmm2['startprob'][1:]))

    transmat = np.zeros((num_states_concat, num_states_concat))
    transmat[:num_states_hmm1+1, :num_states_hmm1+1] = hmm1['transmat']
    transmat[num_states_hmm1:, num_states_hmm1:] = hmm2['transmat']

    means = np.concatenate((hmm1['means'], hmm2['means']), axis=0)

    covars = np.concatenate((hmm1['covars'], hmm2['covars']), axis=0)

    concatenated_hmm = {'startprob': startprob,
                       'transmat': transmat,
                       'means': means,
                       'covars': covars}
    
    return concatenated_hmm


# already implemented, uses concatTwoHMMs()
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]

    for idx in range(1, len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])

    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M emitting (!!) states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M emitting states in the model
    """
    nr_frames, nr_em_states = log_emlik.shape

    ### computation of forward path probabilities (use provided recursion formulas)
    log_alpha = np.empty(shape=(nr_frames, nr_em_states))

    # initialisation (frame / timestep 0)
    log_alpha[0] = log_startprob[:nr_em_states] + log_emlik[0]
    
    # recursion
    for frame_idx in range(1, nr_frames):
        for next_state_idx in range(nr_em_states):
            log_alpha[frame_idx, next_state_idx] = (
                logsumexp(log_alpha[frame_idx - 1] + log_transmat[:nr_em_states, next_state_idx]) + log_emlik[frame_idx, next_state_idx]
            )

    return log_alpha


def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    # very similar to forward except for recursive max (instead of sum) and backtracing housekeeping
    nr_frames, nr_em_states= log_emlik.shape

    ### computation of forward path probabilities (use provided recursion formulas)
    log_v = np.zeros((nr_frames, nr_em_states))
    bt = np.zeros((nr_frames, nr_em_states), dtype=int)

    # initialisation (frame / timestep 0)
    log_v[0] = log_startprob[:nr_em_states] + log_emlik[0]

    # recursion
    for current_frame_idx in range(1, nr_frames):
        for next_state_idx in range(nr_em_states):
            next_state_scores = log_v[current_frame_idx-1] + log_transmat[:nr_em_states, next_state_idx]

            bt[current_frame_idx, next_state_idx] = np.argmax(next_state_scores)
            log_v[current_frame_idx, next_state_idx] = next_state_scores[bt[current_frame_idx, next_state_idx]] + log_emlik[current_frame_idx, next_state_idx]


    # Termination: force ending in the last emitting state, or pick the best
    if forceFinalState:
        last_state = nr_em_states - 1
    else:
        last_state = int(np.argmax(log_v[-1]))

    viterbi_loglik = log_v[-1, last_state]

    # Backtracking
    viterbi_path = np.zeros(nr_frames, dtype=int)
    viterbi_path[-1] = last_state

    for t in range(nr_frames-1, 0, -1):     # no predecessor for frame 0
        viterbi_path[t-1] = bt[t, viterbi_path[t]]

    return viterbi_loglik, viterbi_path


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

    # very similar to forward except that now we walk backwards instead of forward
    nr_frames, nr_em_states = log_emlik.shape

    ### computation of backward path probabilities (use provided recursion formulas)
    log_beta = np.empty(shape=(nr_frames, nr_em_states))

    # initialisation (frame / timestep nr_frames-1)
    log_beta[-1] = 0

    # recursion
    for frame_idx in range(nr_frames-2, -1, -1):
        for state_idx in range(nr_em_states): 
            log_beta[frame_idx, state_idx] = logsumexp(
                log_transmat[state_idx, :nr_em_states] + log_emlik[frame_idx + 1] + log_beta[frame_idx + 1]
            )

    return log_beta


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

    log_px = logsumexp(log_alpha[-1])
    log_gamma = log_alpha + log_beta - log_px

    return log_gamma

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    gamma = np.exp(log_gamma)
    gamma_sum = gamma.sum(axis=0)

    # means[j] = sum_n gamma[n,j] * X[n] / sum_n gamma[n,j]
    means  = (gamma.T @ X) / gamma_sum[:, None]

    # covars[j, d] = sum_n gamma[n,j] * (X[n,d] - means[j,d])^2 / sum_n gamma[n,j]
    covars = np.zeros_like(means)
    for j in range(means.shape[0]):
        diff = X - means[j]
        covars[j] = (gamma[:, j][:, None] * diff**2).sum(axis=0) / gamma_sum[j]

    # variance flooring for numeric stability
    covars = np.maximum(covars, varianceFloor)

    return means, covars