import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing



class BinomialGLM():
    def __init__(self,lambda_, num_trials):
        self.num_trials = num_trials
        self.lambda_ = lambda_

    def set_prox_grad(self, max_iter, tol, lr, rho):
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.rho = rho
        
    def _logit_fun(self, X_train):
        eta = np.matmul(X_train, self.beta_) + self.beta0_
        return 1 / (1 + np.exp(-eta))

    def prox_opt(self, v, mu):
        res = np.zeros(len(v))
        for i in range(len(res)):
            res[i] = abs(v[i]) - mu
            if res[i] <= 0:
                res[i] = 0
            res[i] *= np.sign(v[i])
        return res

    def _loss_fun(self, X_train, Y_train):
        eta = np.matmul(X_train, self.beta_) + self.beta0_
        res = self.num_trials * np.log(1 + np.exp(-eta)) + (self.num_trials - Y_train) * eta
        return res.mean() + self.lambda_ * np.linalg.norm(self.beta_, ord=1)

    def _gradient(self, X_train, Y_train):
        res = Y_train - self.num_trials * self._logit_fun(X_train)
        self._grad_beta0 = - res.mean()
        self._grad_beta = - np.matmul(X_train.T, res) / len(Y_train)

    def fit(self, X_train, Y_train ):
        self.beta0_ = 0.0
        self.beta_ = np.zeros(X_train.shape[1])
        self.eta = self.lr
        

        self.losses = []
        self.losses.append(self._loss_fun(X_train, Y_train))

        for t in range(self.max_iter):
            self._gradient(X_train, Y_train)
            self.beta0_ = self.beta0_ - self.lr  * self._grad_beta0
            beta_new_pre = self.beta_ - self.lr  * self._grad_beta
            self.beta_ = self.prox_opt(beta_new_pre, self.eta *self.lambda_)
            
            self.losses.append(self._loss_fun(X_train, Y_train))
            update = (self.losses[-2] - self.losses[-1]) / abs(self.losses[-2])
            if update <= self.tol:
                break

            self.eta *= self.rho

        self.iter_happened = t + 1

    def get_support(self, thresh):
        self.supp = []
        for i in range(len(self.beta_)):
            if abs(self.beta_[i]) > thresh:
                self.supp.append(i)

    def get_support_by_order(self, num_nb, thresh):
        self.row_norms = abs(self.beta_)
        sorted_index = list(np.argsort(self.row_norms)[::-1])
        self.supp = []

        for i in range(num_nb):
            if self.row_norms[sorted_index[i]] > thresh:
                self.supp.append(sorted_index[i])
            else:
                break


class CondBinomialGLM():
    """
    The class of conditional Binomial GLM.
    """
    def __init__(self, num_trials):
        self.num_trials = num_trials

    def set_prox_grad(self, max_iter, tol, lr, rho):
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.rho = rho

    def _loss_fun(self, X_train, Y_train, lambda_, B_, A_):
        eta = np.matmul(X_train, B_[1:, :]) + B_[0, :]
        tmp = self.num_trials * np.log(1 + np.exp(-eta)) + (self.num_trials - np.expand_dims(Y_train, axis=1)) * eta
        tmp = tmp * A_
        return tmp.sum() / self.n_clusters + lambda_ * self._regu_fun(B_)

    def _regu_fun(self, B_):
        res = 0.0
        for i in range(1,B_.shape[0]):
            res += np.linalg.norm(B_[i, :])   # group lasso here should be the norm on row!!!
        return res

    def _gradient(self, X_train, Y_train, B_, A_):
        eta = np.matmul(X_train, B_[1:, :]) + B_[0, :]
        W_ = np.expand_dims(Y_train, axis=1) - self.num_trials * 1 / (1 + np.exp(-eta))
        self.B_grad_ = np.zeros((X_train.shape[1]+1, self.n_clusters))
        C_tmp = A_ * W_
        self.B_grad_[1:, :] = - np.matmul(X_train.T, C_tmp) / self.n_clusters       # correct for general coefficient gradient
        self.B_grad_[0, :] = - np.matmul(np.ones(X_train.shape[0]), C_tmp) / self.n_clusters  # correct for intercept gradient
    
    def prox_opt(self, B, mu):
    	#mu: threshold for prox to zero
        for j in range(1, B.shape[0]):
            if np.linalg.norm(B[j, :]) <= mu:
                B[j, :] = 0
            else:
                B[j, :] = (1 - mu / np.linalg.norm(B[j, :])) * B[j, :]
        return B
    
    #def cv(self, X, Y, e, kfold, n_clusters, bandwidth, cv_j,thresh,dec,eps = 0.001, K=100, step_3 = False, step_1=True):
    def cv(self, X, Y, e, kfold, n_clusters, bandwidth, eps = 0.001, K=100, step_3 =False):
    # cross validation we need to get lambda.1se and se for each lambda
        
        # calculate lambda_ls
		# lambda_max is analog to logistic group lasso and lambda_min = 0.001 lambda_max
        # lambda is uniform distributed on log scale
        # we should correct here by s(df_g) = sqrt(d.f._g)
        self.n_clusters = n_clusters
        self.bandwidth = bandwidth
        
        # weight kernel matrix
        glob_A_ = np.zeros((X.shape[0], self.n_clusters))
        glob_kmeans = KMeans(n_clusters=self.n_clusters, random_state=123).fit(e)
        for i in range(X.shape[0]):
            for j in range(self.n_clusters):
                glob_A_[i, j] = np.linalg.norm(e[i, :] - glob_kmeans.cluster_centers_[j, :])
        glob_A_ = np.exp(- glob_A_ / bandwidth)
        for j in range(self.n_clusters):
            glob_A_[:, j] = glob_A_[:, j] / glob_A_[:, j].sum()
        #lambda_ls
        self.C_ = np.zeros((X.shape[1],self.n_clusters))
        y_bar = np.mean(Y)
        for i in range(X.shape[1]):
            for j in range(self.n_clusters):
                self.C_[i,j] = np.sum(glob_A_[:,j].reshape((glob_A_.shape[0],1)) * 
                (X[:,i].reshape((X.shape[0],1)) * Y.reshape((Y.shape[0],1)) - y_bar))
        cand_lambda = np.zeros((X.shape[1],))
        for i in range(X.shape[1]):
            cand_lambda[i] = np.linalg.norm(self.C_[i,:]) / 1
        # sqrt(df) np.sqrt(self.n_clusters)
        lambda_max = np.max(cand_lambda)
        lambda_min = eps * lambda_max
        log_ls = np.log(lambda_min) + np.arange(0,K + 1) * np.log(1/eps)/K
        #log_ls = lambda_min + np.arange(0,K + 1) * np.log(1/eps)/K       
        lambda_ls = np.exp(log_ls)
        # should check here 1000
        self.lambda_ls = lambda_ls
        
        #output table
        mse_table = pd.DataFrame(np.zeros((len(self.lambda_ls),kfold+1)),columns=['lambda',*range(kfold)])
        norm_table = pd.DataFrame(np.zeros((len(self.lambda_ls),kfold+1)),columns=['lambda',*range(kfold)])
        out_table = pd.DataFrame(np.zeros((len(self.lambda_ls),4)),columns=['lambda','mse_mean','se','norm'])
        mse_table.loc[:,'lambda'] = self.lambda_ls
        norm_table.loc[:,'lambda'] = self.lambda_ls
        out_table.loc[:,'lambda'] = self.lambda_ls
        
        #cv partition
        num_cores = multiprocessing.cpu_count()
        ind = np.arange(X.shape[0]).reshape((X.shape[0],1)) % kfold
        new_X = pd.DataFrame(np.concatenate((ind,X),axis=1))
        new_Y = pd.DataFrame(np.concatenate((ind,Y.reshape((Y.shape[0],1))),axis = 1))
        new_e = pd.DataFrame(np.concatenate((ind,e),axis=1))
        
        #parallel computation
        results = Parallel(n_jobs = num_cores)(delayed(self.parallel_)(i, new_X,new_Y,new_e,n_clusters,bandwidth) for i in range(kfold))
        j = 0
        for res in results:
            mse_ls, norm_ls = res
            mse_table.loc[:,j] = mse_ls
            norm_table.loc[:,j] = norm_ls
            j += 1
        
        out_table.loc[:,'mse_mean']=np.mean(mse_table.iloc[:,1:],axis=1)
        out_table.loc[:,'se'] = np.sqrt(np.var(mse_table.iloc[:,1:],axis=1))/np.sqrt(kfold)
        out_table.loc[:,'norm'] = np.mean(norm_table.iloc[:,1:], axis = 1)
        # results table is out_table
        lambda_min = out_table.loc[np.argmin(out_table.loc[:,'mse_mean']),'lambda']
        min_se = out_table.loc[np.argmin(out_table.loc[:,'mse_mean']),'se']
        mse_min = np.min(out_table.loc[:,'mse_mean'])
        lambda_1se = np.max(out_table.loc[out_table.loc[:,'mse_mean'] < (mse_min + min_se),'lambda'])
        
        return (lambda_min, lambda_1se)
        
        #raise Exception("Check here for the mse")

    def parallel_(self, i, new_X, new_Y, new_e, n_clusters, bandwidth):
        X_val = np.array(new_X.loc[new_X.loc[:,0]==i,1:])
        Y_val = np.array(new_Y.loc[new_Y.loc[:,0]==i,1:]).flatten()
        e_val = np.array(new_e.loc[new_e.loc[:,0]==i,1:])
        X_train = np.array(new_X.loc[new_X.loc[:,0]!=i,1:])
        Y_train = np.array(new_Y.loc[new_Y.loc[:,0]!=i,1:]).flatten()
        e_train = np.array(new_e.loc[new_e.loc[:,0]!=i,1:])
        B_ls, norm_ls, kmeans = self.fit(X_train,Y_train,e_train, n_clusters, bandwidth)
        mse_ls = self.validate(X_val, Y_val, e_val, B_ls, kmeans)
        return (mse_ls, norm_ls)
    
    
    def fit(self, X_train, Y_train, e_train, n_clusters, bandwidth):
    #self.B_ is weight matrix for regression and the 1st row self.B_[0,:] is intercept term
        B_ls = []
        norm_ls = []
        iter_ls = []
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=123).fit(e_train)
        # Compute the matrix A kernel weight matrix
        A_ = np.zeros((X_train.shape[0], self.n_clusters))
        for i in range(X_train.shape[0]):
            for j in range(self.n_clusters):
                A_[i, j] = np.linalg.norm(e_train[i, :] - kmeans.cluster_centers_[j, :])
        A_ = np.exp(- A_ / bandwidth)
        for j in range(self.n_clusters):
            A_[:, j] = A_[:, j] / A_[:, j].sum()

        # Intialize B
        for lambda_ in self.lambda_ls:
            B_ = np.zeros((X_train.shape[1]+1, self.n_clusters))
        
            eta = self.lr

            losses = []
            losses.append(self._loss_fun(X_train, Y_train, lambda_, B_, A_))

            for t in range(self.max_iter):
                self._gradient(X_train, Y_train, B_, A_)
                B_ = B_ - eta * self.B_grad_
                B_ = self.prox_opt(B_, eta*lambda_)   # should be B_ here
            
                losses.append(self._loss_fun(X_train, Y_train, lambda_, B_, A_))
                update_ = (losses[-2] - losses[-1]) / abs(losses[-2])
                if update_ <= self.tol:
                    break

                eta *= self.rho

            iter_happened = t + 1
            B_ls.append(B_)
            iter_ls.append(iter_happened)
            nonzero_ind = 0
            i_ls = []
            for i in range(1,B_.shape[0]):
                if np.linalg.norm(B_[i, :]) != 0:
                    nonzero_ind += 1
                    i_ls.append(i)
            #if nonzero_ind == 4:
            #    print(i_ls)
            norm_ls.append(nonzero_ind)
        return(B_ls, norm_ls, kmeans)


    def fit2(self, X, Y, e, lambda_, n_clusters, bandwidth):
    # this function fit the model by given lambda_
        self.n_clusters = n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=123).fit(e)
        A_ = np.zeros((X.shape[0], n_clusters))
        for i in range(X.shape[0]):
            for j in range(n_clusters):
                A_[i, j] = np.linalg.norm(e[i, :] - kmeans.cluster_centers_[j, :])
        A_ = np.exp(- A_ / bandwidth)
        for j in range(n_clusters):
            A_[:, j] = A_[:, j] / A_[:, j].sum()
        
        B_ = np.zeros((X.shape[1]+1, n_clusters))
        eta = self.lr
        losses = []
        losses.append(self._loss_fun(X, Y, lambda_, B_, A_))

        for t in range(self.max_iter):
            self._gradient(X, Y, B_, A_)
            B_ = B_ - eta * self.B_grad_
            B_ = self.prox_opt(B_, eta*lambda_)   # should be B_ here
            losses.append(self._loss_fun(X, Y, lambda_, B_, A_))
            
            update_ = (losses[-2] - losses[-1]) / abs(losses[-2])
            if update_ <= self.tol:
                break

            eta *= self.rho
        
        iter_happened = t + 1
        
        return(B_, iter_happened)

    def fit3(self, X_train, Y_train, e_train, n_clusters, bandwidth, seed, gap,thresh):
    #self.B_ is weight matrix for regression and the 1st row self.B_[0,:] is intercept term
        B_return = []
        norm_ls = []
        iter_ls = []
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=123).fit(e_train)
        # Compute the matrix A kernel weight matrix
        A_ = np.zeros((X_train.shape[0], self.n_clusters))
        for i in range(X_train.shape[0]):
            for j in range(self.n_clusters):
                A_[i, j] = np.linalg.norm(e_train[i, :] - kmeans.cluster_centers_[j, :])
        A_ = np.exp(- A_ / bandwidth)
        for j in range(self.n_clusters):
            A_[:, j] = A_[:, j] / A_[:, j].sum()

        # Intialize B
        init = 0
        while True:
            lambda_ = np.exp(np.log(seed) + init * gap)

            B_ = np.zeros((X_train.shape[1]+1, self.n_clusters))        
            eta = self.lr

            losses = []
            losses.append(self._loss_fun(X_train, Y_train, lambda_, B_, A_))

            for t in range(self.max_iter):
                self._gradient(X_train, Y_train, B_, A_)
                B_ = B_ - eta * self.B_grad_
                B_ = self.prox_opt(B_, eta*lambda_)   # should be B_ here
            
                losses.append(self._loss_fun(X_train, Y_train, lambda_, B_, A_))
            
                update_ = (losses[-2] - losses[-1]) / abs(losses[-2])
                if update_ <= self.tol:
                    break

                eta *= self.rho

            iter_happened = t + 1
            #B_return.append(B_)
            iter_ls.append(iter_happened)
            nonzero_ind = 0
            i_ls = []
            for i in range(1,B_.shape[0]):
                if np.linalg.norm(B_[i, :]) != 0:
                    nonzero_ind += 1
                    i_ls.append(i)
            
            if nonzero_ind <= thresh:
                #print(nonzero_ind)
                break
            init += 1
        return(B_, kmeans)

        
    def e_train_cls(self, X_train, Y_train, e_train, cls):
        class_ind = np.zeros((e_train.shape[0],))
        for i in range(e_train.shape[0]):
            ind = np.argmin(np.sum(( e_train[i,:]  - self.kmeans.cluster_centers_)**2,axis=1))
            class_ind[i] = ind
        error = np.sum((cls - class_ind)**2)/cls.shape[0]
        print("the cls error for e_train is: ",error,"which should be a small value")
    
    
    def validate(self, X_val, Y_val, e_val, B_ls, kmeans):
    # this function helps to choose out the best lambda_ which reach min mse in validate set
    # e_val should be a N_val * 3 feature array
        ls_lambda = self.lambda_ls
        class_index = np.zeros((X_val.shape[0],))
        for i in range(X_val.shape[0]):
            ind = np.argmin(np.sum(( e_val[i,:]  - kmeans.cluster_centers_)**2,axis=1))
            class_index[i] = ind
        mse_ls = []
        for B_ in B_ls:
            pred_y = np.zeros((X_val.shape[0],))
            eta = np.matmul(X_val, B_[1:, :]) + B_[0, :]
            for i in range(X_val.shape[0]):
                prob = 1 / (1 + np.exp(-eta[i, int(class_index[i]) ] ))
                pred_y[i] = self.num_trials * prob
            mse = np.mean((Y_val - pred_y)**2)
            mse_ls.append(mse)
        
        return mse_ls
        
    

    def get_support_by_order(self, num_nb, thresh):
    # need to change it to B_ls version
        self.row_norms_ls = []
        self.supp_ls = []
        for B_ in self.B_ls:
            row_norms = np.linalg.norm(B_[1:, :], axis=1)
            self.row_norms_ls.append(row_norms)
            sorted_index = list(np.argsort(row_norms)[::-1])
            supp = []

            for i in range(num_nb):
                if row_norms[sorted_index[i]] > thresh:
                    supp.append(sorted_index[i])
                else:
                    break
            self.supp_ls.append(supp)

def simulate_binomial(N, n, d, s):
    X_train = np.random.binomial(4, 0.1, size=(n, d))
    beta_true = np.zeros(d+1)
    beta_true[:s+1] = np.random.uniform(0.5, 1.0, size=s+1)
    eta = np.matmul(X_train, beta_true[1:]) + beta_true[0]
    prob = 1 / (1 + np.exp(-eta))
    Y_train = np.random.binomial(N, prob)

    return X_train, Y_train, beta_true

def simulate_cond_binomial(N, n, dx, de, s):
    cls = np.random.choice(a=(0, 1), size=n)
    
    X_train = np.random.binomial(4, 0.1, size=(n, dx))
    
    beta_true = np.zeros((dx+1, 2))
    #beta_true[0, :] = 0.0
    beta_true[0, :] = 0.0
    beta_true[:s+1, 0] = np.random.uniform(0.2, 0.5, size=s+1)
    beta_true[:s+1, 1] = np.random.uniform(-0.5, -0.2, size=s+1)

    eta = np.matmul(X_train, beta_true[1:, :]) + beta_true[0, :]

    e_centers = np.array([-3, 3])
    
    Y_train = np.zeros(n)
    e_train = np.zeros((n, de))
    for i in range(n):
        prob = 1 / (1 + np.exp(-eta[i, cls[i]]))
        Y_train[i] = np.random.binomial(N, prob)
        e_train[i, :] = np.random.normal(loc=e_centers[cls[i]], scale=0.01, size=de) #scale=0.5

    return X_train, Y_train, e_train, beta_true, cls


