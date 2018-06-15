import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeans class (filename: kmeans.py')
        #initializing J to 0
        J = 0
        #initialize means
        #picking self.n_cluster from N data points
        # print('x',x)
        # print('shuffled',x)
        # print('n clusters',self.n_cluster)
        centroids = x[np.random.randint(N, size=self.n_cluster)]
        # print('initializing centroids: ',centroids)
        membership = np.zeros((N,self.n_cluster))
        
        for run in range(self.max_iter):
            num_runs = run
            # print('iterations',num_runs)
            for i in range(N):
                # print('centroids',centroids)
                # print('x[i]',x[i])
                diff = centroids - x[i]
                # print('diff',diff)
                dot = np.dot(diff,diff.T)
                # print('dot',dot)
                diags = np.diag(dot)
                # print('diags',diags)
                membership[i][np.argmin(diags)] = 1
            # print(membership)
            
            #calculating Jnew
            Jnew = 0
            for i in range(N):
                idx = np.argmax(membership[i])
                diff = centroids[idx] - x[i]
                Jnew += np.dot(diff.T,diff)
                # print('Jnew',Jnew)
            Jnew = Jnew/N
            # print('Jnew',Jnew)
            # print('J',J)

            #comparing J to Jnew
            if abs(J - Jnew) <= self.e:
                break

            J = Jnew

            #update means
            for i in range(self.n_cluster):
                # centroids[i] = 
                idx = np.where(membership[:,i] == 1)
                # print('idx',idx[0])
                # print('x of idx',x[idx].shape,x[idx])
                centroids[i] = np.sum(x[idx],axis = 0) / x[idx].shape[0]
                # print('centroids[i]',centroids[i])
        # print('membership',membership.shape,membership)
        membership_k = np.zeros(N)
        for i in range(N):
            membership_k[i] = np.argmax(membership[i])
        return centroids, membership_k, num_runs

        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeansClassifier class (filename: kmeans.py')

        #find the centroids
        clf = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, num_runs = clf.fit(x)
        print('centroids: ',centroids.shape,centroids )
        print('N,D',N,D)
        print('membership',membership.shape,membership)
        #assign labels to centroids
        centroid_labels = np.zeros(self.n_cluster)
        for i in range(self.n_cluster):
            idx = np.where(membership == i)
            counts = np.bincount(y[idx])
            centroid_labels[i] = np.argmax(counts)
        print('centroid labels',centroid_labels)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement predict function in KMeansClassifier class (filename: kmeans.py')

        preds = np.zeros((N))
        for i in range(N):
            diff = self.centroids - x[i]
            dot = np.dot(diff,diff.T)
            diags = np.diag(dot)
            preds[i] = self.centroid_labels[np.argmin(diags)]
        return preds
        # DONOT CHANGE CODE BELOW THIS LINE
