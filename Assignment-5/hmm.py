from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  # computing values for each time-step
  for t in range(N):
    # base case for t == 0
    if t == 0:
        alpha[:,0] = np.multiply(pi,B[:,0])
        # print('alpha[:,0] = ',alpha[:,0])
        # print('alpha[:,0]',alpha[:,0])
    else:
        for j in range(S):
        # alpha[:,t] = B[:,t] * np.dot(A[:,t-1].T,alpha[:,t-1])
        # alpha[:,t] = alpha[:,t] / np.sum(alpha[:,t])
            # print('A',A.shape)
            # print('B',B.shape)
            # print('mult shapes',B[:,O[t]].shape,alpha[:,t-1].shape)
            # print('mult',np.multiply(B[:,O[t]], alpha[:,t-1]).shape)
            # print('dot',np.dot(np.multiply(B[:,O[t]], alpha[:,t-1]).T, A).shape)
            # alpha[:,t] = np.dot(np.multiply(B[:,O[t]], alpha[:,t-1]), A)
            alpha[j,t] = B[j,O[t]] * np.sum(np.multiply(alpha[:,t-1],A[:,j]))
        # print('alpha[:,',t,'] = ', alpha[:,t])
        # print('alpha[:,t]',alpha[:,t])
  # print('sum of alpha[:,5] is ',np.sum(alpha[:,5]))
  # print('alpha',alpha)
  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  
  #computing beta values by going back in time from T-1 to 0
  for t in range(N-1,-1,-1):
    if t == (N-1):
        beta[:,t] = 1
    else:
        for i in range(S):
        # print('hi')
            for j in range(S):
                # mult = np.dot(np.multiply(beta[:,t+1],B[:,O[t+1]]).T,A)
                # beta[:,t] = mult.T
                beta[i,t] += beta[j,t+1] * A[i,j] * B[j,O[t+1]]
  # print('beta',beta)
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  prob = np.sum(alpha[:,-1])

  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  prob = np.sum(np.multiply(np.multiply(beta[:,0],pi),B[:,O[0]]))

  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  ###################################################
  T = len(O)
  K = A.shape[0]
  #creating two tables, one for the probabilities of the most likely path, and one for the states of the most likely path
  P = np.zeros((K,T))
  S = np.zeros((K,T))


  #computing the most likely path at time t
  for t in range(T):
    if t == 0:
        P[:,0] = np.multiply(pi,B[:,0])
        # print('P[:,0] = ',P[:,0])
        # print('starting P',P)
    else:
        for j in range(K):
            # print('thing to max: ',np.multiply(P[:,t-1],A[:,j]) * B[j,O[t]])
            # print('max,argmax',np.max(np.multiply(P[:,t-1],A[:,j]) * B[j,O[t]]),int(np.argmax(np.multiply(P[:,t-1],A[:,j]) * B[j,O[t]])))
            # print('P[:,',t,'] = ',np.multiply(P[:,t-1],A[:,j]) * B[j,O[t]], '-- max ', np.max(np.multiply(P[:,t-1],A[:,j]) * B[j,O[t]]))
            P[j,t] = np.max(np.multiply(P[:,t-1],A[:,j]) * B[j,O[t]])
            # print('did thing to max change?1 ',np.multiply(P[:,t-1],A[:,j]) * B[j,O[t]])
            S[j,t] = int(np.argmax(np.multiply(P[:,t-1],A[:,j]) * B[j,O[t]]))
            # print('did thing to max change? ',np.multiply(P[:,t-1],A[:,j]) * B[j,O[t]])
            # print('argmax again',int(np.argmax(np.multiply(P[:,t-1],A[:,j]) * B[j,O[t]])))
            # print('S[j,t]',S[j,t])
            # print('P',P)
            # print('S',S)
  # print('S',S)
  #state of most likely observation at the end
  path.append(int(np.argmax(P[:,T-1])))
   
  #going back in time and finding the most likely state at t
  for t in range(T-2,-1,-1):
    # print('last value of path',path[-1])
    path.append(int(S[path[-1],t]))
  # print('path',path)
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()