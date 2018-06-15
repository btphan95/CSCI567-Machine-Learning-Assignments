import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None
        self.R = None
    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k using k-means')
            #initialize using k-means clustering
            clf = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means, membership, num_runs = clf.fit(x)
            #find gammas from membership
            gammas = membership

            #compute variance and pi_k
            self.variances = np.zeros((self.n_cluster,D,D))
            self.pi_k = np.zeros(self.n_cluster)
            for k in range(self.n_cluster):
                idx = np.where(gammas == k)
                mat = np.zeros((D,D))
                for ind in idx[0]:
                    diff = x[ind] - self.means[k]
                    diff = diff.reshape((D,1))
                    # print('diff',diff.shape)
                    # print('dot',np.dot(diff,diff.T))
                    mat += np.dot(diff,diff.T)
                count = idx[0].shape[0]
                self.variances[k] = mat/count
                self.pi_k[k] = count / N
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k randomly')

            #initialize means randomly
            self.means = np.random.rand(self.n_cluster,D)

            #variances consist of identity matrices, and pi_k's are 1/K
            self.variances = np.zeros((self.n_cluster,D,D))
            self.pi_k = np.zeros(self.n_cluster)
            for k in range(self.n_cluster):
                self.variances[k] = np.identity(D)
                self.pi_k[k] = 1/self.n_cluster
            
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement fit function (filename: gmm.py)')
        #initializing responsibilities
        print('x',x.shape)
        log_likelihood = 0
        R = np.zeros((N,self.n_cluster))
        for k in range(self.n_cluster):
            try:
                if np.linalg.det(self.variances[k]) == 0:
                    # print('det was 0')
                    self.variances[k] = self.variances[k] + np.identity(D)*.003
                    if np.linalg.det(self.variances[k]) == 0:
                        self.variances[k] = self.variances[k] + np.identity(D)*.003
            except:
                try:
                    # print('det was 0')
                    self.variances[k] = self.variances[k] + np.identity(D)*.003
                    np.linalg.det(self.variances[k]) == 0
                except:
                    # print('det was 0')
                    self.variances[k] = self.variances[k] + np.identity(D)*.003
            # print('det',np.linalg.det(self.variances[k]))
            diff = x - self.means[k]
            exp = -.5 * np.dot(np.dot(diff,np.linalg.inv(self.variances[k])),diff.T)
            diag = np.diag(exp)
            gaussian = np.linalg.det(self.variances[k]) ** -.5 ** (2 * np.pi) ** (-D/2.) * np.exp(diag)
            R[:,k] = self.pi_k[k] * gaussian
    

        for run in range(self.max_iter):
            print('iteration ',run)
            num_runs = run
            R = (R.T / np.sum(R, axis = 1)).T
            N_k = np.sum(R, axis = 0)
            N_k = np.sum(R,axis=0)
            
            #M-step: updating parameters
            for k in range(self.n_cluster):
                self.means[k] = np.dot(x.T,R[:, k]) / N_k[k]
                diff = x - self.means[k]
                self.variances[k] = np.dot(diff.T,np.multiply(diff,R[:,k].reshape((N,1)))) / N_k[k]
                self.pi_k[k] = N_k[k] / N

            #E-step: updating responsibilities
                try:
                    if np.linalg.det(self.variances[k]) == 0:
                        # print('det was 0')
                        self.variances[k] = self.variances[k] + np.identity(D)*.003
                        if np.linalg.det(self.variances[k]) == 0:
                            self.variances[k] = self.variances[k] + np.identity(D)*.003
                except:
                    try:
                        # print('det was 0')
                        self.variances[k] = self.variances[k] + np.identity(D)*.003
                        np.linalg.det(self.variances[k]) == 0
                    except:
                        # print('det was 0')
                        self.variances[k] = self.variances[k] + np.identity(D)*.003
                # print('flag')
                diff = x - self.means[k]
                # print('diff',diff[:5])
                exp = -.5 * np.dot(np.dot(diff,np.linalg.inv(self.variances[k])),diff.T)
                diag = np.diag(exp)
                # print('before',diag[:5])
                # print('after',np.exp(diag[:5]))
                gaussian = np.linalg.det(self.variances[k]) ** -.5 ** (2 * np.pi) ** (-D/2.) * np.exp(diag)
                # print('after',gaussian[:5])
                R[:,k] = self.pi_k[k] * gaussian
                # print('error here? R',sum(R[:,:k]))
            #compute log-likelihood
            new_ll = np.sum(np.log(np.sum(R, axis = 1)))
            # print('log_likelihood',new_ll)
            if abs(new_ll - log_likelihood) < self.e:
                break
            log_likelihood = new_ll
        return num_runs
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement sample function in gmm.py')

        #create random array
        # print('N',N)
        # print('D',self.n_cluster)
        samples = np.zeros((N,self.means.shape[1]))
        for i in range(N):
            # print('pi_k',self.pi_k)
            k = np.argmax(np.random.multinomial(1,self.pi_k,1))
            # print('k',k)
            # print('samples[i]',samples[i])
            # print('means shape',self.means[k].shape)
            # print('variance shape',self.variances[k].shape)
            # print('samples',np.random.multivariate_normal(self.means[k],self.variances[k]))
            samples[i] = np.random.multivariate_normal(self.means[k],self.variances[k])
            # print('samples[i]',samples[i])
        return samples
        # print('samples',samples.shape,samples[0])
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement compute_log_likelihood function in gmm.py')
        N,D = x.shape
        R = np.zeros((N,self.n_cluster))
        for k in range(self.n_cluster):
            try:
                if np.linalg.det(self.variances[k]) == 0:
                    self.variances[k] = self.variances[k] + np.identity(D)*.003
                    if np.linalg.det(self.variances[k]) == 0:
                        self.variances[k] = self.variances[k] + np.identity(D)*.003
            except:
                try:
                    # print('det was 0')
                    self.variances[k] = self.variances[k] + np.identity(D)*.003
                    np.linalg.det(self.variances[k]) == 0
                except:
                    # print('det was 0')
                    self.variances[k] = self.variances[k] + np.identity(D)*.003
            diff = x - self.means[k]
            exp = -.5 * np.dot(np.dot(diff,np.linalg.inv(self.variances[k])),diff.T)
            diag = np.diag(exp)
            gaussian = np.linalg.det(self.variances[k]) ** -.5 ** (2 * np.pi) ** (-D/2.) * np.exp(diag)
            R[:,k] = self.pi_k[k] * gaussian
    

        #compute log-likelihood
        log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
        # print(type(float(log_likelihood)))
        return float(log_likelihood)
        # DONOT MODIFY CODE BELOW THIS LINE
