# three steps of personalized dag learning

import numpy as np
import pandas as pd
from BinomialGML_cv import BinomialGLM, CondBinomialGLM
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import multiprocessing

class GraphEmbedSimu():
    """
    The class of graph embedding simulation, we simulate a graph and node feature
    then estimate matrix B to reduce dimension of node feature.
    
    Attributes:
        num_users: X's row number
        num_features: X's col number should be a large number. around 100
        a : control sparsity of graph
        C_com : b = C_com * a community effect
        C_coef: para should be a vector if r > 1, dim: r * 1
    
    """
    def __init__(self, num_users, num_features, a, C_coef, C_com, nonlinear):
        self.num_users = num_users
        self.num_features = num_features
        self.a = a
        self.C_coef = C_coef
        self.C_com = C_com
        self.nonlinear = nonlinear
        
    def simulation(self, r):
        """ simulation X and graph.
            Attributes:
                r : dimension of B0
        """
        self.B0 = np.zeros((self.num_features,r))
        self.B0[0,:] = 1
        self.B0[1,:] = 1
        self.Sigma = np.zeros((self.num_features, self.num_features))
        for t1 in range(self.num_features):
            for t2 in range(self.num_features):
                if abs(t1 - t2) < 5:
                    self.Sigma[t1,t2] = 0.4**(abs(t1 - t2))

        node_labels = np.random.binomial(n = 1, p = 0.5, size = self.num_users)
        self.node_labels = node_labels
        self.X = np.zeros((self.num_users,self.num_features))
        for ind in range(self.num_users):
            if self.node_labels[ind] == 0:
                Xi = np.random.multivariate_normal(np.pi*np.ones((self.num_features,)), self.Sigma, size = 1)
            else:
                Xi = np.random.multivariate_normal(-np.pi*np.ones((self.num_features,)), self.Sigma, size = 1)
            self.X[ind,:] = Xi
        
        if nonlinear:
            self.X = np.sin(self.X)
        self.W = np.zeros((self.num_users,self.num_users))  # undirected graph
        
        for i in range(self.num_users):
            for j in range(i,self.num_users):
                prod = np.matmul(self.B0.T,(self.X[i,:] - self.X[j,:]).reshape((self.num_features,1)))
                
                if node_labels[i] == node_labels[j]: # Ci = Cj  P(wij | Ci, Cj) = a
                    prob = self.a * np.exp(1 - self.C_coef.dot(np.abs(prod))) / (1 + np.exp(1 - self.C_coef.dot(np.abs(prod))))
                    
                else:
                    prob = self.C_com * self.a * np.exp(1 - self.C_coef.dot(np.abs(prod))) / (1 + np.exp(1 - self.C_coef.dot(np.abs(prod))))
                prob = prob.flatten()
                self.W[i,j] = np.random.binomial(n = 1, p = prob, size = 1)
                if self.W[i,j] == 1:
                    self.W[j,i] = 1
        np.save(save_pth + 'UserNet.npy', self.W)
        np.save(save_pth + 'UserLab.npy', self.node_labels) 
        np.save(save_pth + 'UserFeat.npy', self.X) 
        
        

    def estimation(self,r):
        """
        estimate B
        r : dim of B 'p*r'
        """
        self.S = 1 - self.W
        G = np.zeros((self.num_features, self.num_features))
        for i in range(self.num_users):
            for j in range(self.num_users):
                if i != j:
                    Xij = self.X[i,:] - self.X[j,:]
                    G = G + self.S[i,j]*np.matmul(Xij.reshape((self.num_features,1)),Xij.reshape((1,self.num_features)))
        
        G = G/(self.num_users * (self.num_users - 1))
        A_inv = np.linalg.inv(self.Sigma)
        evalues, evectors = np.linalg.eig(A_inv)
        assert (evalues >= 0).all()
        A_inv_half = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)
        w, v = np.linalg.eigh(A_inv_half @ G @ A_inv_half)
        phi = np.zeros((self.num_features, r))
        for ind in range(r):
            phi[:,ind] = v[:,self.num_features - (ind + 1)]
        B = A_inv_half.dot(phi)
        #print(B)
        
        cov_low = np.matmul(self.X, B.reshape((self.num_features,r)))
        labels = KMeans(n_clusters=2, random_state=413).fit(cov_low).labels_
        #print(labels)
        print(sum(labels==1))
        print("error rate for labeling",np.sum(np.abs(self.node_labels - labels))/self.num_users)
        return(cov_low)
        ind_dict = {}
        for lab_ind in range(len(labels)):
            lab = labels[lab_ind]
            if lab not in ind_dict.keys():
                ind_dict[lab] = [lab_ind]
            else:
                ind_dict[lab].append(lab_ind)

        return ind_dict
    


class RelationNetwork():
    """
    The class of relationship network, used to
    help learn the causal DAG.

    Attributes:
        num_nodes: An integer indicating the number of nodes.
        adj_matrix: A numpy 2-dimensional matrix encoding the 
                    adjacency relationship between nodes
        node_feature: A numpy matrix with shape num_nodes x dim of feature. 
                      The raw node feature of user, such as age, sex, etc.
        embd_feature: A numpy matrix with shape num_nodes x dim of embbeed feature.
                      The embbed node feature used for DAG learning
    """
    def __init__(self, n):
        """
        Initialization function.

        Params:
           n: An integer indicating the sample size or number of nodes
        """
        self.num_nodes = n
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.node_feature = None
        self.embd_feature = None

    def simulate_ER(self, prob, dim_feat):
        """
        The method function to simulate a synthetic relationship network
        by Erdos-Renyi Model. The node feature is generated by i.i.d.
        Gaussian random varaiables. Calling the method function will fill in 
        adj_matrix and node_feature with simulated data.

        Params:
           prob: A number in [0,1] indicating the probability of presence
                 for any edge between any two nodes. A larger prob imply
                 a denser graph.
           dim_feat: The dimension of raw feature.
        """
        self.adj_matrix = np.random.choice(a=(0,1), size=(self.num_nodes, self.num_nodes), p=(1-prob, prob))
        for i in range(self.num_nodes):
            self.adj_matrix[i, i] = 0
            for j in range(i):
                self.adj_matrix[j, i] = self.adj_matrix[i, j]
        
        self.node_feature = np.random.normal(loc=0.0, scale=0.1, size=(self.num_nodes, dim_feat))

    def linear_embed(self, embd_mat):
        """
        The method function to do linear embedding. For each node, the function
        generates its embedded feature by adding the linear transformed raw
        features of its neighbors. The function will fill in self.embd_feature.

        Params:
           embd_mat: A numpy matrix as the embedding matrix, with the shape as
                     dim of embbeed feature x dim of feature.
        """
        dim_embd = embd_mat.shape[0]  # The dimension of embedded feature
        # Transform every node feature by embedding matrix
        pre_embd_mat = np.matmul(self.node_feature, embd_mat.T)
        # Compute the embedded feature
        self.embd_feature = np.zeros((self.num_nodes, dim_embd))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adj_matrix[i, j] == 1:
                    # Sum over all neighbors
                    self.embd_feature[i, :] += pre_embd_mat[j, :] # should be [j,:] here

    def linear_embed_simulate(self, dim_embd):
        """
        The method function to do linear embedding by generating embedding matrix
        via i.i.d. Gaussian entries. Encode feature by Gaussian matrix

        Params:
           dim_embd: The dimension of embedded feature
        """
        dim_feat = self.node_feature.shape[1]  # The dimension of raw feature.
        # Generate the embedding matrix with i.i.d. normal
        self.embd_mat = np.random.normal(loc=0.0, scale=1.0, size=(dim_embd, dim_feat))
        self.linear_embed(self.embd_mat)




class BinomialDAGSimulate():
    """
    The class of Binomial DAG generator based on input covariates.
    The order is determined by random shuffling.
    The edge is then generated by Erdos Renyi model.
    Finally, the data is generated via a generalized linear model for Binomial response
    based on the DAG structure.

    Attributes:
        num_nodes: The number of nodes of the DAG
        ordering: The odering of the DAG. A dictionary with key being the order and
                  value being the node index. For example, {0:1, 1:0, 2:2} means that
                  the ordering is 1->0->2.
        adj_matrix: A numpy 2-dimensional matrix encoding the DAG structure
        covarites: The covariates to generate data.
        num_trials: The number of trials of binomial distribution
    """
    def __init__(self, num_nodes, covariates, max_parents, num_trials, save_path):
        """
        Initalization function.

        Params:
           num_nodes: Number of nodes of the DAG.
           covariates: The covariates to generate data.
                       A numpy matrix with the shape of
                       number of samples x dim of embedded feature
           max_parents: The maximum number of parents  (should be 3 here)
           num_trials: The number of trials of binomial distribution
        """
        self.num_nodes = num_nodes
        self.ordering = None
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.moral_graph = np.zeros((self.num_nodes, self.num_nodes))
        self.covariates = covariates
        self.max_parents = max_parents
        self.num_trials = num_trials
        self.save_path = save_path

    def simulate_structure(self):
        """
        The method to generate the DAG structure.
        The order is first determined by random shuffling.
        The edge is then generated by Erdos Renyi model.
        The adajecent ordering nodes are always connected.
        Nodes with closer ordering nodes have the priority to be chosen.
        The number of parents for each node cannot exceed max_parents.
        Check Section 4 of https://www.jmlr.org/papers/volume18/17-243/17-243.pdf

        Params:
           prob: The probability of Erdos Renyi model.
        """
        # We first decide the ordering by random shuffling.
        pre_ordering = np.arange(self.num_nodes)
        np.random.shuffle(pre_ordering)
        self.ordering = pre_ordering

        # We then determine the causal structure based on
        # the ordering via Erodos Renyi model. The adajecent ordering
        # nodes are always connected. Nodes with closer ordering nodes
        # have the priority to be chosen.
        for j in range(1, self.num_nodes):
            self.adj_matrix[self.ordering[j-1], self.ordering[j]] = 1

        for j in range(self.num_nodes-1, 1, -1):
            chosen_parentes = np.random.choice(self.ordering[:j], size=self.max_parents-1, replace=False)
            chosen_parentes = list(chosen_parentes)
            for i in chosen_parentes:
                self.adj_matrix[i, self.ordering[j]] = 1

        for i in range(self.num_nodes):
            for j in range(self.num_nodes): # error here we need iterate all adj_matrix
                if self.adj_matrix[i, j] == 1:
                    self.moral_graph[i, j] = 1
                    self.moral_graph[j, i] = 1
        print("order:",self.ordering)
        print("adj:",self.adj_matrix)
        print("moral:",self.moral_graph)

    def simulate_hetero_data(self):
        """
        The method to generate heterogenous observations based on causal
        DAG structure and also the covariates.
        The weight function is set to be linear, that is,
        w_{lj}(z_embd) = intercept + <w_{lj}, z_mebd>.
        The nonzero parameters are generated uniformly random 
        in the range [0.01 0.05].

        params:
           N: The toal number of trials of Binomial distribution
        """
        # The weight function is set to be linear. We generate
        # parameters by i.i.d. normal and record them as
        # self.weight_params
        sample_size, dim_cov = self.covariates.shape
        #self.weight_params = np.random.uniform(low=0.01, high=0.05, size=(self.num_nodes, self.num_nodes, dim_cov+1))
        
        #self.weight_params = np.random.uniform(low= -0.5, high= -0.25, size=(self.num_nodes, self.num_nodes, dim_cov+1))
        self_weight = np.random.uniform(low=0.1,high = 0.2,size = 1)
        weights_1 = np.random.uniform(low = -1.0,high = -0.5, size = (self.num_nodes,self.num_nodes))
        weights_2 = np.random.uniform(low = 0.5, high = 1, size = (self.num_nodes, self.num_nodes))

        self.observations = np.zeros((sample_size, self.num_nodes))  # the simulated observations
        for i in range(sample_size):
            if i < 2500:
                self.weight_params_new = weights_1
            else:
                self.weight_params_new = weights_2
            for j in range(self.num_nodes):  # The j-th node in ordering
                # Compute the weight vectors
                #weights = np.matmul(self.weight_params[:, self.ordering[j], 1:], self.covariates[i, :])\
                #+ self.weight_params[:, self.ordering[j], 0]
                if j == 0:
                    prob = 1/ (1 + np.exp(-self_weight))
                    self.observations[i, self.ordering[j]] = np.random.binomial(self.num_trials, prob)
                    continue
                
                # Compute the probability for binomial distribution
                #eta = weights[self.ordering[j]]*0.1  # node_j self weight
                eta = 0
                #self.weights_mat[i,:] = weights
                #if i % 1000 == 0:
                    #print(self.weight_params_new)
                for l in range(j):  # We only need to consider the parents of j
                    eta += self.weight_params_new[self.ordering[l],self.ordering[j]] * self.observations[i, self.ordering[l]] *\
                        self.adj_matrix[self.ordering[l],self.ordering[j]]
                prob = 1 / (1 + np.exp(-eta))
                # Generate the data
                self.observations[i, self.ordering[j]] = np.random.binomial(self.num_trials, prob)
        
        pd.DataFrame(self.observations).to_csv(self.save_path+"obs.csv",header=None, index=None)
        pd.DataFrame(self.adj_matrix).to_csv(self.save_path+"adj.csv",header=None, index=None)
        pd.DataFrame(self.moral_graph).to_csv(self.save_path+"moral.csv",header=None, index=None)
                
        return None
        #raise Exception("save file successfully")

class HomoBinomialDAGSimulate():
    """
    The class of Homogeneous Binomial DAG generator based on input covariates.
    The order is determined by random shuffling.
    The edge is then generated by Erdos Renyi model.
    Finally, the data is generated via a generalized linear model 
    for Binomial response based on the DAG structure.

    Attributes:
        num_nodes: The number of nodes of the DAG
        ordering: The odering of the DAG. A dictionary with key being the order and
                  value being the node index. For example, {0:1, 1:0, 2:2} means that
                  the ordering is 1->0->2.
        adj_matrix: A numpy 2-dimensional matrix encoding the DAG structure
        num_trials: The number of trials of binomial distribution
    """
    def __init__(self, num_nodes, max_parents, num_trials):
        """
        Initalization function.

        Params:
           num_nodes: Number of nodes of the DAG.
           max_parents: The maximum number of parents
           num_trials: The number of trials of binomial distribution
        """
        self.num_nodes = num_nodes
        self.ordering = None
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.moral_graph = np.zeros((self.num_nodes, self.num_nodes))
        self.max_parents = max_parents
        self.num_trials = num_trials

    def simulate_structure(self):
        """
        The method to generate the DAG structure.
        The order is first determined by random shuffling.
        The edge is then generated by Erdos Renyi model.
        The adajecent ordering nodes are always connected.
        Nodes with closer ordering nodes have the priority to be chosen.
        The number of parents for each node cannot exceed max_parents.
        Check Section 4 of https://www.jmlr.org/papers/volume18/17-243/17-243.pdf
        """
        # We first decide the ordering by random shuffling.
        pre_ordering = np.arange(self.num_nodes)
        np.random.shuffle(pre_ordering)
        self.ordering = {i: pre_ordering[i] for i in range(self.num_nodes)}

        self._ordering_to_lst()

        # We then determine the causal structure based on
        # the ordering via Erodos Renyi model. The adajecent ordering
        # nodes are always connected. Nodes with closer ordering nodes
        # have the priority to be chosen.
        for j in range(1, self.num_nodes):
            self.adj_matrix[self.ordering[j-1], self.ordering[j]] = 1
        
        for j in range(self.num_nodes-1, 1, -1):
            chosen_parentes = np.random.choice(self.ordering_lst[:(j-1)], size=self.max_parents-1, replace=False)
            chosen_parentes = list(chosen_parentes)
            for i in chosen_parentes:
                self.adj_matrix[i, self.ordering[j]] = 1

        self.moral_graph = self.adj_matrix + self.adj_matrix.T

        self._parents_to_dict()

        self.weight_params = np.zeros((self.num_nodes, self.num_nodes))
        self.weight_params[self.ordering[0], self.ordering[0]] = np.random.uniform(low=0.1, high=0.2, size=1)
        for j in range(self.num_nodes):
            self.weight_params[self.parent_dict[j], j] = np.random.uniform(low=0.5, high=1.0, size=len(self.parent_dict[j])) / self.num_trials

    def _parents_to_dict(self):
        """
        Use a dictionary to store all parents information.
        """
        self.parent_dict = {}
        for j in range(self.num_nodes):
            self.parent_dict[j] = []
            for l in range(self.num_nodes):
                if self.adj_matrix[l, j] == 1:
                    self.parent_dict[j].append(l)

    def _ordering_to_lst(self):
        self.ordering_lst = []
        for key in range(self.num_nodes):
            self.ordering_lst.append(self.ordering[key])

    def simulate_observations(self, sample_size):
        """
        The method to generate homogeneousgeneous observations based on causal
        DAG structure. The nonzero parameters are generated uniformly
        random in the range [0.01 0.05].

        params:
           N: The toal number of trials of Binomial distribution.
        """
        self.observations = np.zeros((sample_size, self.num_nodes))  # the simulated observations
        for j in range(self.num_nodes):  # The j-th node in ordering
            if j == 0:
                eta = self.weight_params[self.ordering[j], self.ordering[j]]
                prob = 1 / (1 + np.exp(-eta))
                self.observations[:, self.ordering[j]] = np.random.binomial(n=self.num_trials, p=prob, size=sample_size)
                continue
            
            eta = self.weight_params[self.ordering[j], self.ordering[j]] * np.ones(sample_size)
            eta += np.matmul(self.observations[:, self.parent_dict[self.ordering[j]]], self.weight_params[self.parent_dict[self.ordering[j]], self.ordering[j]])
            prob = 1 / (1 + np.exp(-eta))
            self.observations[:, self.ordering[j]] = np.random.binomial(n=self.num_trials, p=prob, size=sample_size)

class HomoBinomialDAGLearner():
    """
    The class of Homogeneous Binomial DAG Model
    """
    def __init__(self, num_nodes, num_trials, max_parents, max_nbs):
        """
        Initilization function.

        Params:
           num_nodes: The number of nodes.
           num_trials: The number of trials of Binomial distribution.
           max_parents: Maximum number of parents allowed.
           max_nbs: Maximum number of neighbors allowed.
        """
        self.num_nodes = num_nodes
        self.num_trials = num_trials
        self.max_parents = max_parents
        self.max_nbs = max_nbs

        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.moral_graph = np.zeros((self.num_nodes, self.num_nodes))
        self.skeleton = np.zeros((self.num_nodes, self.num_nodes))

        self.ordering = []
    
    def learn_dag(self, observations, lambda_, thresh):
        """
        The function to learn a causal dag from observations.

        Params:
           observations: An sample size x num_nodes numpy array.
           lambda_: The regularization parameter for lasso
           thresh: The threshold to ensure that the conditioned sets
                   to have enough samples.
        """
        print("Start to learn the DAG!")

        if observations.shape[1] != self.num_nodes:
            raise ValueError("The number of nodes of the object: {} is inconsistent with the observations: {}!".format(self.num_nodes, observations.shape[1]))

        self.lambda_ = lambda_
        self.thresh = thresh
        

        self._learn_dag_step1(observations)
        print("Step 1 is finished!")

        self._learn_dag_step2(observations)
        print("Step 2 is finished!")

        self._learn_dag_step3(observations)
        print("Step 3 is finished!")

        self.skeleton = self.adj_matrix + self.adj_matrix.T
    
    def _learn_dag_step1(self, observations):
        """
        The first step to learn a causal dag from observations,
        that is, to learn a moral graph.

        Params:
           observations: An sample size x num_nodes numpy array.
        """
        # Iterate over each node need to update to parallel
        num_cores = multiprocessing.cpu_count()
        
        #Parallel(n_jobs = num_cores)(delayed(self.parallel_1)
        for j in range(self.num_nodes):
            Y_train = observations[:, j]  # Set y as j-th node
            tmp_index = [i for i in range(self.num_nodes)]
            tmp_index = tmp_index[:j] + tmp_index[(j+1):]
            X_train = observations[:, tmp_index]
            
            # create a binomial glm object to fit the data
            bin_glm = BinomialGLM(lambda_=self.lambda_, num_trials=self.num_trials)
            # set up the parameters for optimization algorithm, that is, proximal gradient method
            bin_glm.set_prox_grad(max_iter=5000, tol=1e-9, lr=1e-4, rho=1.0)
            # fit the data
            bin_glm.fit(X_train, Y_train)
            # hard threshold to get the support
            bin_glm.get_support_by_order(self.max_nbs, 1e-4)

            # Recover the neighborhood of node j
            for i in bin_glm.supp:
                self.moral_graph[tmp_index[i], j] = 1
                self.moral_graph[j, tmp_index[i]] = 1

    def parallel_1(self, i, obs):
        pass

    
    def _learn_dag_step2(self, observations):
        """
        The second step to learn a causal dag from observations,
        that is, to decide the ordering.

        Params:
           observations: An sample size x num_nodes numpy array.
           thresh: The threshold to ensure that the conditioned sets to have enough samples.
        """
        # decide the cut-off sample size
        thresh_size = np.ceil(observations.shape[0]*self.thresh)
        
        # The score matrix to store ODS scores
        self.scores = np.zeros((self.num_nodes, self.num_nodes))
        
        # decide the root
        self._overdisp_score0(observations)
        self.ordering.append(np.argmin(self.scores[0, :]))
        
        # iterate to decide all the ordering
        for m in range(1, self.num_nodes - 1):
            ancestors = set(self.ordering)
            for j in range(0, self.num_nodes):
                if j in ancestors:
                    self.scores[m, j] = np.Inf
                    continue
                
                cand_parent = set(np.where(self.moral_graph[:,j]==1)[0]) & ancestors
                cand_parent = list(cand_parent)

                self.scores[m, j] = self._overdisp_score(observations, thresh_size, j, cand_parent)
            
            self.ordering.append(np.argmin(self.scores[m, :]))
        last_node = list(set(np.arange(self.num_nodes)) - set(self.ordering))[0]
        self.ordering.append(last_node)
        print("The estimated order is:++++++++++++")
        print(self.ordering)

    def _learn_dag_step3(self, observations):
        """
        The third step to learn a causal dag from observations,
        that is, to learn the directed edges based on ordering.

        Params:
           observations: An sample size x num_nodes numpy array.
        """
        # Iterate over each node
        # start from the node with smaller order
        for j in range(self.num_nodes):  # node with j-th order
            # the root node has no parent
            if j == 0:
                continue
            elif j == 1:
                Y_train = observations[:,self.ordering[j]]
                X_train = observations[:,[self.ordering[0],self.ordering[0]]]
                bin_glm = BinomialGLM(lambda_=np.log(self.num_nodes)/40, num_trials=self.num_trials)
                bin_glm.set_prox_grad(max_iter=5000, tol=1e-9, lr=1e-4, rho=1.0)
                bin_glm.fit(X_train, Y_train)
                bin_glm.get_support_by_order(len(self.ordering[:j]), 1e-4)
                if len(bin_glm.supp)>0:
                    self.adj_matrix[self.ordering[0],self.ordering[1]] = 1
                continue

            else:
            # Set y as the node with j-th order
                Y_train = observations[:, self.ordering[j]]
             # Set X as the other nodes with orders smaller than j
                X_train =  observations[:, self.ordering[:j]]
                 
            # we need cross-validation here to select the best lambda
            # Create a dictionary to map the index of X to original dag index
            index_dict = {}
            for i in range(j):
                index_dict[i] = self.ordering[i]
            
            # create a binomial glm object to fit the data
            bin_glm = BinomialGLM(lambda_=self.lambda_, num_trials=self.num_trials)
            # set up the parameters for optimization algorithm, that is, proximal gradient method
            bin_glm.set_prox_grad(max_iter=5000, tol=1e-9, lr=1e-4, rho=1.0)
            # fit the data by model
            bin_glm.fit(X_train, Y_train)
            # hard threshold to get the support
            bin_glm.get_support_by_order(len(self.ordering[:j]), 1e-4)

            # Recover the parent nodes for j-th ordered node
            for i in bin_glm.supp:
                self.adj_matrix[index_dict[i], self.ordering[j]] = 1    

    def _overdisp_score(self, observations, thresh_size, j, supp):
        """
        The overdispersion score function for m!=0.
        """
        tmp_dict = self._cond_mean_var_helper(observations, j, supp)
        sample_size_lst = []
        ods_score_lst = []
        for key in tmp_dict:
            if sum(np.asarray(tmp_dict[key])>=1)<=1:
                continue
            sample_size_lst.append(len(tmp_dict[key]))
            # add a small value to avoid numerical error
            omega = 1 / (1 - np.mean(tmp_dict[key]) / self.num_trials - 1e-20)
            
            ods_score = (omega**2) * np.var(tmp_dict[key]) - omega * np.mean(tmp_dict[key])
            ods_score_lst.append(ods_score)
        
        weights = np.array(sample_size_lst)
        weights = weights / weights.sum()
        ods_scores = np.array(ods_score_lst)
        
        return (weights * ods_scores).sum()

    def _cond_mean_var_helper(self, observations, j, supp):
        """
        The function to help compute the conditional mean and variance.

        Param:
           j: The index of interest.
           supp: The list of indices to be conditioned on.
        """
        if j in supp:
            raise ValueError("j cannot in support!")
        
        # Creat a dictionary, where the key is the raw python bytes string
        # of the numpy arrays belonging to supp and the value is the list
        # of values of node j with the corresponding condition value.
        tmp_dict = {}
        for i in range(observations.shape[0]):
            if observations[i, supp].tobytes() in tmp_dict:
                tmp_dict[observations[i, supp].tobytes()].append(observations[i, j])
            else:
                tmp_dict[observations[i, supp].tobytes()] = []
                tmp_dict[observations[i, supp].tobytes()].append(observations[i, j])

        return tmp_dict
    
    def _overdisp_score0(self, observations):
        """
        The overdispersion score function for m=0.
        """
        mean_v = observations.mean(0)
        var_v = observations.var(0)
        # add a small value to avoid numerical error
        omega_v = 1 / (1 - mean_v / self.num_trials - 1e-20)
        self.scores[0, :] = (omega_v**2) * var_v - omega_v * mean_v



class CondiBinomialDAGLearner():
    """
    The class of Conditional Binomial DAG Model with covariates
    """ 
   
    def __init__(self, num_nodes, num_trials, max_parents, max_nbs):
        """
        Initialization function
        
        Params:
            num_nodes: the number of nodes. [p]
            num_trials:  the number of trials of Binomial
            max_parents: Maximum number of parents allowed
            max_nbs: Maximum number of neighbors allowed
        """
        self.num_nodes = num_nodes
        self.num_trials = num_trials
        self.max_parents = max_parents
        self.max_nbs = max_nbs
       
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.moral_graph = np.zeros((self.num_nodes, self.num_nodes))
        #self.moral_graph here is CandidateParents in R code
                
        self.skeleton = np.zeros((self.num_nodes, self.num_nodes))
       
        self.ordering = []
   
    def learn_dag(self, obs, e, sparsity_level, thresh, n_clusters, bandwidth, w):
        """
        The function is to learn a casual dag from observations.
       
        Params:
            obs: sample size(N) * num_nodes numpy array.
            e  : feature / covariates for each person
            thresh: the threshold to ensure the conditioned sets to
        have enough samples.
        """
        print("Start to learn DAG!")
        if obs.shape[1] != self.num_nodes:
            raise ValueError("The number of nodes of the object: {} is inconsistent with the observations: {    }!".\
            format(self.num_nodes, obs.shape[1]))
        
        self.thresh = thresh
        self.n_clusters = n_clusters
        self._learn_dag_step1(obs,e, sparsity_level)
        print("Step 1 finished")
        
        self._learn_dag_step2(obs, e, bandwidth, score_weights = w)
        print("Step 2 finished")
        
        self._learn_dag_step3(obs, e, sparsity_level)
        print("Step 3 finished")
        
        self.skeleton = self.adj_matrix + self.adj_matrix.T
        # skeleton is undirected graph
        
    def _learn_dag_step1(self, obs, e, sparsity_level):
        """
        The first step is to learn the moral graph
        
        Params:
            obs: an sample size (N) * num_nodes (p) numpy array
        """
        for j in range(self.num_nodes):
            Y = obs[:, j]
            tmp_ind = [i for i in range(self.num_nodes)]
            tmp_ind = tmp_ind[:j] + tmp_ind[(j+1):]
            X = obs[:, tmp_ind]
            
            cond_bin_glm = CondBinomialGLM(num_trials = self.num_trials)
            # need to check lr and lambda_ls
            cond_bin_glm.set_prox_grad(max_iter = 5000, tol = 1e-7, lr = 0.05, rho=1)
            # need to check n_clusters
            (lambda_min, lambda_1se) = cond_bin_glm.cv(X,Y,e, kfold = 20, n_clusters = self.n_clusters, bandwidth = 0.1)
            opt = lambda_min + sparsity_level * (lambda_1se - lambda_min)
            (B_, iter_num) = cond_bin_glm.fit2(X, Y, e, lambda_ = opt, n_clusters = self.n_clusters, bandwidth = 0.1)
            # B_ is the coeff estimated
            for i in range(1,B_.shape[0]):
                if np.linalg.norm(B_[i, :]) != 0:
                    self.moral_graph[tmp_ind[i-1], j] = 1
                    self.moral_graph[j, tmp_ind[i-1]] = 1  
            print(self.moral_graph)
            #raise Exception("Check moral graph here, it should be symmetric")
            print("We have found the neighbors for:",j)


    def _learn_dag_step2(self, obs, e, bandwidth, score_weights):
        """Scoring method for ordering
        self.ordering will be updated in this step
        """
	    
	    # the score matrix
        self.scores = np.zeros((self.num_nodes, self.num_nodes))
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=123).fit(e)
        A_ = np.zeros((obs.shape[0], self.n_clusters))
        for i in range(obs.shape[0]):
            for j in range(self.n_clusters):
                A_[i, j] = np.linalg.norm(e[i, :] - kmeans.cluster_centers_[j, :])
        A_ = np.exp(- A_ / bandwidth)
        for j in range(self.n_clusters):
            A_[:, j] = A_[:, j] / A_[:, j].sum()
        
        E_1 = np.matmul(obs.T, A_)  # should be around self.num_trials
        #print(E_1)
        #raise Exception("Check here for E1 matrix")
        
        
        V_1 = np.zeros((E_1.shape[0],E_1.shape[1]))
        for i in range(self.n_clusters):
            m1 = np.square(obs - np.repeat(E_1[:,i].reshape((1,E_1.shape[0])),obs.shape[0],axis=0))
            V_1[:,i] = np.matmul(m1.T, A_[:,i].reshape((obs.shape[0],1))).flatten()
        
        w_1 = 1/(1 - E_1/self.num_trials - 1e-20)
        if np.any(w_1 < 0):
            raise Exception("error for w_1")
        sepe_score = w_1**2 * V_1 - w_1 * E_1
        sc_1 = sepe_score.dot(score_weights.reshape((1,len(score_weights))).T)

        self.scores[0,:] = sc_1.flatten()
        self.ordering.append(np.argmin(sc_1)) # first node
        for m in range(1, self.num_nodes - 1):
            anc = set(self.ordering)
            for j in range(0,self.num_nodes):
                if j in anc:
                    self.scores[m,j] = np.Inf
                    continue
                cand_pa = set(np.where(self.moral_graph[:,j]==1)[0]) & anc
                cand_pa = list(cand_pa)
                if j in cand_pa:
                    raise ValueError("j cannot in support!")
                tmp_dict = {} # key: X_pa; val: [index for this key]
                for i in range(obs.shape[0]):
                    if obs[i, cand_pa].tobytes() in tmp_dict:
                        tmp_dict[obs[i,cand_pa].tobytes()].append(i)
                    else:
                        tmp_dict[obs[i,cand_pa].tobytes()] = []
                        tmp_dict[obs[i,cand_pa].tobytes()].append(i)
                sample_size_ls = []
                ods_score_ls = []
                for key in tmp_dict:
        	        if sum(obs[tmp_dict[key],j]>=1)<=1:
        	            continue
        	        sample_size_ls.append(len(tmp_dict[key]))
        	        tmp_A = A_[tmp_dict[key],:]
        	        for ind_j in range(self.n_clusters):
        	            tmp_A[:,ind_j] = tmp_A[:,ind_j] / tmp_A[:,ind_j].sum()
        	        E_2 = np.matmul(obs[tmp_dict[key], j].T, tmp_A)
        	        
        	        V_2 = np.zeros((1,E_2.shape[0]))
        	        for ind_i in range(self.n_clusters):
        	            m2 = np.square(obs[tmp_dict[key],j] - E_2[ind_i])
        	            V_2[0,ind_i] = np.matmul(m2.T, tmp_A[:,ind_i])
        	        V_2 = V_2.flatten()
        	        w_2 = 1/(1 - E_2 / self.num_trials - 1e-20)
        	        sepe_score = w_2**2 * V_2 - w_2 * E_2
        	        sc_2 = sepe_score.dot(score_weights.reshape((1,len(score_weights))).T)
        	        ods_score_ls.append(sc_2)
        	    #now we have the score ls for each key
                weights = np.array(sample_size_ls)
                weights = weights/ weights.sum()
                ods_scores = np.array(ods_score_ls)
                self.scores[m,j] = (weights * ods_scores).sum()
                
            self.ordering.append(np.argmin(self.scores[m,:]))
        last_node = list(set(np.arange(self.num_nodes)) - set(self.ordering))[0]
        self.ordering.append(last_node)
        print(self.ordering)
        
        	        
	    
	    
    def _learn_dag_step3(self,obs,e, sparsity_level):
        """DAG estimation
        """
        for j in range(4,self.num_nodes):
            print("Now we estimate parents node for node:",j)
            if j == 0:
                continue

            elif j == 1:
                Y = obs[:, self.ordering[j]] # node 2
                X = obs[:, [self.ordering[0],self.ordering[0]]]
                cond_bin_glm = CondBinomialGLM(num_trials = self.num_trials)
                cond_bin_glm.set_prox_grad(max_iter = 5000, tol = 1e-7, lr = 0.05, rho =1.0)
                # why self.num_nodes/40
                (B_, iter_num) = cond_bin_glm.fit2(X,Y, e, lambda_ = np.log(self.num_nodes)/40, n_clusters = self.n_clusters, bandwidth=0.1)
                for norm_i in range(1,B_.shape[0]):
                    if np.linalg.norm(B_[norm_i, :]) > 0:
                        self.adj_matrix[self.ordering[0],self.ordering[1]] = 1
                continue
            else:
                Y = obs[:,self.ordering[j]] # node j to be the response
                X = obs[:,self.ordering[:j]] # nodes before node j
            ind_dict = {}
            for i in range(j):
                ind_dict[i] = self.ordering[i]
	        
            cond_bin_glm = CondBinomialGLM(num_trials = self.num_trials)
            cond_bin_glm.set_prox_grad(max_iter = 5000, tol = 1e-7, lr = 1e-3, rho=1)
            (lambda_min, lambda_1se) = cond_bin_glm.cv(X,Y,e, kfold = 20, n_clusters = self.n_clusters, bandwidth = 0.1)

            opt = lambda_min + sparsity_level * (lambda_1se - lambda_min)
            (B_, iter_num ) = cond_bin_glm.fit2(X, Y, e, opt, n_clusters = self.n_clusters, bandwidth = 0.1)
            #(B_, iter_num ) = cond_bin_glm.fit2(X, Y, e, self.lambda_ls_[j-4], n_clusters = self.n_clusters, bandwidth = 0.1)
            for i in range(1,B_.shape[0]):
                if np.linalg.norm(B_[i, :]) != 0:
                    self.adj_matrix[ind_dict[i - 1], self.ordering[j]] = 1
            print(self.adj_matrix)



class DAGEvaluate():
    """
    The class to evalue the performance of DAG algorithm
    """
    def __init__(self, dag_true, dag_learned):
        """
        The initialization function.

        Params:
           dag_true: The true dag.
           dag_learned: The learned dag. 
        """
        self.dag_true = dag_true
        self.dag_learned = dag_learned

    def eval_ordering(self):
        """
        Evaualte the learned ordering.
        """
        # The relative hamming distance between learned ordering and
        # the true ordering.
        self.ordering_hamming = (np.array(self.dag_true.ordering) \
            != np.array(self.dag_learned.ordering)).sum() / len(self.dag_learned.ordering)
        print("The relative hamming distance of ordering is: {}".format(self.ordering_hamming))

    def eval_moral_graph(self):
        """
        Evaluate the learned moral graph in step 1.
        """
        self.moral_graph_TP = 0
        self.moral_graph_FP = 0
        self.moral_graph_TN = 0
        self.moral_graph_FN = 0

        for i in range(self.dag_true.num_nodes):
            for j in range(self.dag_true.num_nodes):
                if self.dag_true.moral_graph[i, j] == 1 and self.dag_learned.moral_graph[i, j] == 1:
                    self.moral_graph_TP += 1
                elif self.dag_true.moral_graph[i, j] == 0 and self.dag_learned.moral_graph[i, j] == 1:
                    self.moral_graph_FP += 1
                elif self.dag_true.moral_graph[i, j] == 0 and self.dag_learned.moral_graph[i, j] == 0:
                    self.moral_graph_TN += 1
                elif self.dag_true.moral_graph[i, j] == 1 and self.dag_learned.moral_graph[i, j] == 0:
                    self.moral_graph_FN += 1
        
        self.moral_graph_precision = self.moral_graph_TP / (self.moral_graph_TP + self.moral_graph_FP)
        self.moral_graph_recall = self.moral_graph_TP / (self.moral_graph_TP + self.moral_graph_FN)

        print("The preicison of step 1 is: {}".format(self.moral_graph_precision))
        print("The recall of step 1 is: {}".format(self.moral_graph_recall))

    def eval_directed_edges(self):
        """
        Evaluate the learned directed edges.
        """
        # The relative hamming distance between learned directed edges and
        # the true directed edges.
        self.dir_edg_hamming = (self.dag_learned.adj_matrix != self.dag_true.adj_matrix).sum() \
            / (self.dag_learned.num_nodes*(self.dag_learned.num_nodes-1))
        print("The relative hamming distance of directed edges is: {}".format(self.dir_edg_hamming))

    def eval_skeleton(self):
        """
        Evaluate the learned skeleton.
        """
        # The relative hamming distance between learned moral graph and
        # the true moral graph.
        self.skeleton_hamming = (self.dag_learned.skeleton != self.dag_true.moral_graph).sum() \
            / (self.dag_learned.num_nodes*(self.dag_learned.num_nodes-1))
        print("The relative hamming distance of skeleton is: {}".format(self.skeleton_hamming))


if __name__ == "__main__":
    
    save_pth = "./" # Input path for saving results
    
    dim = 1 # dimension after embedding
    num_users = 2000 # number of users in the network
    num_nodes = 10 # number of nodes in the DAG
    seed = 243247 # random seed
    nonlinear = False # nonlinear embedding
    
    np.random.seed(seed)
    
    if nonlinear:
        embed = GraphEmbedSimu(num_users = num_users, num_features = 50, a = 0.99 , C_coef = 3*np.ones(dim,), C_com =0.1,nonlinear=nonlinear)
        embed.simulation(r = dim) # User feature saved in UserFeat.npy
        try:
            cov = np.load("gae/ReducedFeat.npy") # this file exists only after running gae/train.py
        except:
            raise Exception("Please copy the file to gae/ and run the Command 'python gae.train.py'")
    
    else:
        # linear Embedding
        embed = GraphEmbedSimu(num_users = num_users, num_features = 50, a = 0.99 , C_coef = 3*np.ones(dim,), C_com =0.1,nonlinear=nonlinear)
        embed.simulation(r = dim)
        cov = embed.estimation(r = dim)
    
    # If AX = B B:cov   then X could be all 1 and A : 0.2/100 or 0.5/100 if A feature is N*100
    
    #cov = np.ones((5000,2))
    #cov[ind_dict[0],:] = [0.2,0.2]
    #cov[ind_dict[1],:] = [0.5,0.5]
    #raise Exception("Now we have the full simulation pipeline.")
    
    #cov = np.ones((5000,2))
    #cov[:2500,:] = [-0.5,-0.5]
    #cov[2500:,:] = [0.5,0.5]
    
    heto_bin = BinomialDAGSimulate(num_nodes, cov, 3 ,num_trials = 4., save_path=save_pth)
    heto_bin.simulate_structure()
    heto_bin.simulate_hetero_data()
    
    print("Simulation data is generated!")
    
    dag_learned = CondiBinomialDAGLearner(num_nodes = num_nodes, num_trials = 4,max_parents = 4, max_nbs = 1)
    dag_learned.learn_dag(heto_bin.observations,cov, sparsity_level = 2.0, thresh = 1, n_clusters=2, bandwidth=0.1, w = np.array([0.,1]))
    # sparsity_level 1.5-2 / bandwidth

    #dag_true = HomoBinomialDAGSimulate(num_nodes=10, max_parents=2, num_trials=4)
    #dag_true.simulate_structure()
    #dag_true.simulate_observations(10000)

    order_learned = dag_learned.ordering
    adj_matrix_learned = dag_learned.adj_matrix
   
    
    #dag_learned = HomoBinomialDAGLearner(10, 4, 9, 9)
    #regu_param = 1.5 / np.log(max(2500, 10))
    #dag_learned.learn_dag(heto_bin.observations[:2500,:], regu_param, 0.005)

    evaluator = DAGEvaluate(heto_bin, dag_learned)
    evaluator.eval_ordering()
    evaluator.eval_moral_graph()
    evaluator.eval_directed_edges()
    #evaluator.eval_skeleton()


