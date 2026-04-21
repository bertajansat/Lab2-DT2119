import numpy as np
from lab2_tools import *
from prondict import prondict
import matplotlib.pyplot as plt
from lab1_proto import mfcc
import lab1_tools
import time
import copy


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

    Output
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
    num_states_hmm1 = len(hmm1['startprob'])-1
    num_states_hmm2 = len(hmm2['startprob'])-1
    num_states_concat = num_states_hmm1+num_states_hmm2+1
    
    startprob = np.concatenate((hmm1['startprob'],hmm2['startprob'][1:]))

    transmat = np.zeros((num_states_concat,num_states_concat))
    transmat[:num_states_hmm1+1,:num_states_hmm1+1] = hmm1['transmat']
    transmat[num_states_hmm1:,num_states_hmm1:] = hmm2['transmat']

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
    for idx in range(1,len(namelist)):
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
    #TODO ???

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """

    N, M = log_emlik.shape
    logalpha = np.zeros((N, M))

    # Initialization
    logalpha[0] = log_startprob + log_emlik[0]
    # Recursion
    for t in range(1, N):
        for j in range(M):
            logalpha[t, j] = log_emlik[t, j] + logsumexp(
                logalpha[t-1] + log_transmat[:M, j]
            )
    return logalpha




def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    backward_prob = np.zeros((N, M))

    # Initialization
    backward_prob[-1] = 0
    # Recursion (backwards in time)
    for t in range(N-2, -1, -1):
        for j in range(M):
            #backward_prob[t, j] = logsumexp(log_transmat[j, :] +log_emlik[t+1, :] + backward_prob[t+1, :])
            backward_prob[t, j] = logsumexp(log_transmat[j, :M] + log_emlik[t+1, :M] + backward_prob[t+1, :M])
    return backward_prob

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

    N, M = log_emlik.shape
    viterbi_loglik_v = np.zeros((N, M))
    backptr = np.zeros((N, M), dtype=int)

    # Initialization
    viterbi_loglik_v[0] = log_startprob + log_emlik[0]
    # Recursion
    for t in range(1, N):
        for j in range(M):
            viterbi_loglik_v[t, j] = log_emlik[t, j] + np.max(viterbi_loglik_v[t-1]+log_transmat[:M, j])
            backptr[t,j] = np.argmax(viterbi_loglik_v[t-1]+log_transmat[:M, j])

    # Backtracking:
    viterbi_path = np.zeros(N, dtype=int)

    if forceFinalState:
        last_state = M - 1
        viterbi_path[-1] = last_state
        viterbi_loglik = viterbi_loglik_v[-1, last_state]
    else:
        last_state = np.argmax(viterbi_loglik_v[-1])
        viterbi_path[-1] = last_state
        viterbi_loglik = np.max(viterbi_loglik_v[-1])

    # Backtracking
    for t in range(N-2, -1, -1):
        viterbi_path[t] = backptr[t+1, viterbi_path[t+1]]

    return viterbi_loglik, viterbi_path



def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

    log_gamma = log_alpha + log_beta - logsumexp(log_alpha[-1])

    return log_gamma

def gmmPosteriors(log_emlik):
    log_gamma = log_emlik - logsumexp(log_emlik, axis=1).reshape(-1, 1)  # TODO: Revise!!

    
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
    N, D = X.shape
    M = log_gamma.shape[1]
    means = np.zeros((M, D))
    covars = np.zeros((M, D))

    for i in range(M):
        #means[i] = np.sum(np.exp(log_gamma[:,i])[:, None] * X)/np.sum(np.exp(log_gamma[:,i])) # Change log_gamma (N,) to (N,1)
        #covars[i] = np.sum(np.exp(log_gamma[:,i])[:, None]*(X-means[i])**2)/np.sum(np.exp(log_gamma[:,i]))
        means[i] = np.sum(np.exp(log_gamma[:, i])[:, None] * X, axis=0) / np.sum(np.exp(log_gamma[:, i]))      # axis=0 !
        covars[i] = np.sum(np.exp(log_gamma[:, i])[:, None] * (X - means[i])**2, axis=0) / np.sum(np.exp(log_gamma[:, i]))  # axis=0 !
        
        covars[i]=np.maximum(covars[i], varianceFloor)
    return means, covars




## MAIN CODE

data = np.load('lab2_data.npz', allow_pickle=True)['data']

## 4
# Load model files
phoneHMMs = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()

#print(list(sorted(phoneHMMs.keys())))
#print(phoneHMMs['ah'].keys())


isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']

wordHMMs = {}
wordHMMs['o'] = concatHMMs(phoneHMMs, isolated['o'])

# Load example
example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()

## 5.1 

hmm=wordHMMs['o'] 
lpr = log_multivariate_normal_density_diag(example['lmfcc'],hmm['means'],hmm['covars']) # Log probability of each observation X comes from each HMM (for word o) state

comp_equal=np.allclose(lpr, example['obsloglik']) # True if both arrays are almost equal, with a given tolerance

print(f'\n**log_mulivariate_normal_density_diag() and example[obsloglik] give very simmilar results: {comp_equal}')

# log probability of given word under gaussian states of HMM for that digit

for digit in list(sorted(prondict.keys())):

    wordHMMs[digit] = concatHMMs(phoneHMMs, isolated[digit])
    hmm_digit = wordHMMs[digit]

    loglik = log_multivariate_normal_density_diag(
        example['lmfcc'],
        hmm_digit['means'],
        hmm_digit['covars']
    )
    """
    plt.imshow(loglik.T, aspect='auto', origin='lower')
    plt.title(f"Log-likelihood for frame and state (digit {digit})")
    plt.xlabel("Frames")
    plt.ylabel("HMM states")
    plt.colorbar()
    plt.show()   # REVISAR
    """

# 5.2


M = lpr.shape[1]
hmm=wordHMMs['o'] 
forward_logalpha = forward(lpr,np.log(hmm['startprob'][:M]),np.log(hmm['transmat'][:M, :M])) # :M to remove additional states (transition)

comp_equal=np.allclose(forward_logalpha, example['logalpha']) # True if both arrays are almost equal, with a given tolerance

print(f'\n**forward() output and example[logalpha] give very simmilar results: {comp_equal}')


# Likelihood

log_likelihood = logsumexp(forward_logalpha[-1]) # Only alpha_(N-1)
comp_equal=np.allclose(log_likelihood, example['loglik']) # True if both arrays are almost equal, with a given tolerance

print(f'\n**Computed log likelihood ({log_likelihood}) and example[logalik] ({example['loglik']}) give very simmilar results: {comp_equal}. ')



phoneHMMs_all = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item() #Load HMMs with multiple speakers


wordHMMs_all = {}
for digit in list(sorted(prondict.keys())):  # Concatenation to obtain word HMMs with multiple speakers

    wordHMMs_all[digit] = concatHMMs(phoneHMMs_all, isolated[digit])
"""
results_one = []
results_all = []

start_time_forward = time.perf_counter()
for utterance in data:
    log_likelihoods = []
    log_likelihoods_all = []
    words_one = []
    words_all = []
    samples = utterance['samples']
    sr = utterance['samplingrate']
    digit = utterance['digit']
    gender = utterance['gender']
    repetition = utterance['repetition']
    
    lmfcc = mfcc(samples, winlen = int(0.020*sr), winshift = int(0.01*sr), nfft=512, nceps=13)

    for word, model in wordHMMs.items():
        lpr_one=log_multivariate_normal_density_diag(lmfcc,model['means'],model['covars'])
        M = lpr_one.shape[1]
        forward_logalpha_one = forward(lpr_one,np.log(model['startprob'][:M]),np.log(model['transmat'][:M, :M]))
        log_likelihood_one = logsumexp(forward_logalpha_one[-1])
        log_likelihoods.append(log_likelihood_one)
        words_one.append(word)

    best_index = np.argmax(log_likelihoods)
    predicted = words_one[best_index]

    results_one.append((digit,predicted))

    # For HMMs trained with multiple speakers
    for word, model in wordHMMs_all.items():
        lpr_all=log_multivariate_normal_density_diag(lmfcc,model['means'],model['covars'])
        M = lpr_all.shape[1]
        forward_logalpha_all = forward(lpr_all,np.log(model['startprob'][:M]),np.log(model['transmat'][:M, :M]))
        log_likelihood_all = logsumexp(forward_logalpha_all[-1])
        log_likelihoods_all.append(log_likelihood_all)
        words_all.append(word)
    
    best_index_all = np.argmax(log_likelihoods_all)
    predicted_all = words_all[best_index_all]

    results_all.append((digit,predicted_all))

accuracy_one = sum(t==p for t,p in results_one)/len(results_one)
print("\nFORWARD ALGORITHM:")
print(f"\nAccuracy for one speaker HMMs: {accuracy_one}")
accuracy_all = sum(t==p for t,p in results_all)/len(results_all)
print(f"Accuracy for multiple speakers HMMs: {accuracy_all}")

end_time_forward = time.perf_counter()
# Heatmaps

labels = sorted(list(wordHMMs.keys()))
label_to_idx = {l:i for i,l in enumerate(labels)}

cm_one = np.zeros((len(labels),len(labels)))
for t,p in results_one:
    cm_one[label_to_idx[t],label_to_idx[p]] += 1

fig, ax = plt.subplots()

ax.imshow(cm_one, cmap='Blues')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix - Single Speaker')

fig.colorbar(ax.images[0])

plt.show()

labels = sorted(list(wordHMMs_all.keys()))
label_to_idx = {l:i for i,l in enumerate(labels)}

cm_all= np.zeros((len(labels),len(labels)))
for t,p in results_all:
    cm_all[label_to_idx[t],label_to_idx[p]] += 1

fig, ax = plt.subplots()

ax.imshow(cm_all, cmap='Blues')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix - All Speakers')

fig.colorbar(ax.images[0])

plt.show()
"""

# 5.3
M = lpr.shape[1]
hmm=wordHMMs['o'] 
viterbi_loglik, viterbi_best_path = viterbi(lpr,np.log(hmm['startprob'][:M]),np.log(hmm['transmat'][:M, :M]))

comp_equal=np.allclose(viterbi_loglik, example['vloglik']) # True if both arrays are almost equal, with a given tolerance

print(f'\n**Viterbi output and example[vloglik] give very simmilar results: {comp_equal}')

plt.imshow(forward_logalpha.T, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='log alpha')

plt.plot(viterbi_best_path, color='red', linewidth=2)

plt.title("Logalpha including Viterbi path")
plt.xlabel("Time")
plt.ylabel("State")
plt.show()

"""
# Predict 44 utterances
results_one = []
results_all = []
start_time_viterbi = time.perf_counter()
for utterance in data:
    viterbi_loglikes_one = []
    viterbi_loglikes_all = []
    words_one = []
    words_all = []
    samples = utterance['samples']
    sr = utterance['samplingrate']
    digit = utterance['digit']
    gender = utterance['gender']
    repetition = utterance['repetition']
    
    lmfcc = mfcc(samples, winlen = int(0.020*sr), winshift = int(0.01*sr), nfft=512, nceps=13)

    for word, model in wordHMMs.items():
        lpr_one=log_multivariate_normal_density_diag(lmfcc,model['means'],model['covars'])
        M = lpr_one.shape[1]
        viterbi_loglike_one, best_path_one = viterbi(lpr_one,np.log(model['startprob'][:M]),np.log(model['transmat'][:M, :M]))
        viterbi_loglikes_one.append(viterbi_loglike_one)
        words_one.append(word)

    best_index = np.argmax(viterbi_loglikes_one)
    predicted = words_one[best_index]

    results_one.append((digit,predicted))

    # For HMMs trained with multiple speakers
    for word, model in wordHMMs_all.items():
        lpr_all=log_multivariate_normal_density_diag(lmfcc,model['means'],model['covars'])
        M = lpr_all.shape[1]
        viterbi_loglike_all, best_path_all = viterbi(lpr_all,np.log(model['startprob'][:M]),np.log(model['transmat'][:M, :M]))
        viterbi_loglikes_all.append(viterbi_loglike_all)
        words_all.append(word)
    
    best_index_all = np.argmax(viterbi_loglikes_all)
    predicted_all = words_all[best_index_all]

    results_all.append((digit,predicted_all))

accuracy_one = sum(t==p for t,p in results_one)/len(results_one)
print("\nVITERBI:")
print(f"Accuracy for one speaker HMMs: {accuracy_one}")
accuracy_all = sum(t==p for t,p in results_all)/len(results_all)
print(f"Accuracy for multiple speakers HMMs: {accuracy_all}")
end_time_viterbi = time.perf_counter()
# Heatmaps

labels = sorted(list(wordHMMs.keys()))
label_to_idx = {l:i for i,l in enumerate(labels)}

cm_one = np.zeros((len(labels),len(labels)))
for t,p in results_one:
    cm_one[label_to_idx[t],label_to_idx[p]] += 1

fig, ax = plt.subplots()

ax.imshow(cm_one, cmap='Blues')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix - Single Speaker')

fig.colorbar(ax.images[0])

plt.show()

labels = sorted(list(wordHMMs_all.keys()))
label_to_idx = {l:i for i,l in enumerate(labels)}

cm_all= np.zeros((len(labels),len(labels)))
for t,p in results_all:
    cm_all[label_to_idx[t],label_to_idx[p]] += 1

fig, ax = plt.subplots()

ax.imshow(cm_all, cmap='Blues')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix - All Speakers')

fig.colorbar(ax.images[0])

plt.show()


total_forward= end_time_forward-start_time_forward
total_viterbi = end_time_viterbi-start_time_viterbi-end_time_viterbi

print(f"Computational time for forward algorithm: {total_forward}. Computational time for Viterbi algorithm: {total_viterbi}")
"""

## 5.4

M = lpr.shape[1]
hmm=wordHMMs['o'] 
backward_logbeta = backward(lpr,np.log(hmm['startprob'][:M]),np.log(hmm['transmat'][:M, :M]))

comp_equal=np.allclose(backward_logbeta, example['logbeta']) # True if both arrays are almost equal, with a given tolerance

print(f'\n**Backward output and example[logbeta] give very simmilar results: {comp_equal}')

# TODO: Optional part

# 6.1

log_gamma = statePosteriors(forward_logalpha,backward_logbeta)
gamma = np.exp(log_gamma)
print(f"\nSum of state posteriors for each time step: {np.sum(gamma, axis=1)}")

log_gamma_gmm=gmmPosteriors(lpr)
gamma_gmm = np.exp(log_gamma_gmm)

print(f"\nGMM Posteriors: Sum over time for each state: {np.sum(gamma_gmm, axis=1)}")
print(f"Sumer over time and over states: {np.sum(gamma_gmm)}")
print(f"Number of frames: {len(gamma_gmm)}")


# 6.2


for word in wordHMMs_all:
    likelihoods = []
    #model = wordHMMs_all[word]
    model = copy.deepcopy(wordHMMs_all[word]) # Use copy of the model to avoid errors

    samples = data[10]['samples']
    sr = data[10]['samplingrate']
    #model = wordsHMMs_all['4']

    lmfcc = mfcc(samples, winlen = int(0.020*sr), winshift = int(0.01*sr), nfft=512, nceps=13)

    # 1. Log-likelihood of the data:
    lpr_all = log_multivariate_normal_density_diag(lmfcc, model['means'], model['covars'])
    M = model['means'].shape[0]

    iterations = 20
    threshold = 1.0

    # likelihood inicial correcte (forward)
    log_alpha = forward(lpr_all, np.log(model['startprob'][:M]), np.log(model['transmat'][:M, :M]))
    log_likelihood = logsumexp(log_alpha[-1])

    conversion = False

    #print(word)
    #print(data[10]['digit'])
    print(f"Model: {word}, M={M}, log-lik INICIAL: {logsumexp(log_alpha[-1]):.2f}")

    for i in range(iterations):
        # 2. Compute alpha, beta, gamma probs (forward, backward, statePosterior)
        log_alpha = forward(lpr_all, np.log(model['startprob'][:M]), np.log(model['transmat'][:M, :M]))
        log_beta = backward(lpr_all, np.log(model['startprob'][:M]), np.log(model['transmat'][:M, :M]))
        log_gamma = statePosteriors(log_alpha, log_beta)

        #print(f"  iter {i}: log_alpha={log_alpha}, beta={log_beta}, gamma: {log_gamma}")
    
        # 3. Update mean and variance
        mean, covar = updateMeanAndVar(lmfcc, log_gamma)#,varianceFloor=5)

        model['means'] = mean
        model['covars'] = covar

        #print(f"Means: {mean}, covars: {covar}")

        # 4. Estimate new likelihood.
        lpr_new = log_multivariate_normal_density_diag(lmfcc, model['means'] , model['covars'])

        log_alpha_new = forward(lpr_new, np.log(model['startprob'][:M]), np.log(model['transmat'][:M, :M]))
        new_likelihood = logsumexp(log_alpha_new[-1])
        likelihoods.append(new_likelihood)
        
        diff = new_likelihood - log_likelihood
        
        print(f"Likelihood epoch {i}: {new_likelihood}")

        if diff > 0 and diff < threshold:
            log_likelihood = new_likelihood
            lpr_all = lpr_new
            conversion = True
            break
        else:
            log_likelihood = new_likelihood
            lpr_all = lpr_new
    
    plt.plot(likelihoods)
    plt.title(f"Log-likelihood convergence for model {word}")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")

    #plt.ylim(min(likelihoods[1:]), max(likelihoods[1:]))

    plt.show()
        

    print(f"Model: {word}. Conversion: {conversion}. Number of iterations used: {i}. Final log-lik: {log_likelihood:.2f}")





