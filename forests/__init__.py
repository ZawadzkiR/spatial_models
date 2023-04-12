from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class GW_RandomForestRegressor():
  def __init__(self,
               kernel = 'adaptive',
               bw=100,
               ntree = 500,
               mtry = None,
               importance = 'impurity',
               nthreads = None,
               weighted = True,
               print_results = True,
               forests = True
               ):
    """
    Constructor for GW_RandomForestRegressor class.
    :param kernel: str, default 'adaptive'. Type of kernel to use for nearest neighbor selection. Options are 'adaptive' or 'fixed'.
    :param bw: int, default 100. Bandwidth for kernel selection.
    :param ntree: int, default 500. Number of trees to use for random forest.
    :param mtry: int, default None. Number of variables to consider for each split. If None, set to max(int(len(X)/3), 1).
    :param importance: str, default 'impurity'. Type of feature importance measure. Options are 'impurity' or 'permutation'.
    :param nthreads: int, default None. Number of threads to use for random forest.
    :param weighted: bool, default True. Whether to use weighted random forest.
    :param print_results: bool, default True. Whether to print results of kernel selection.
    :param forests: bool, default True. Whether to use random forests in local models.
    """

    self.bw = bw
    self.kernel = kernel
    self.ntree = ntree
    self.mtry = mtry
    self.importance = importance
    self.nthreads = nthreads
    self.forests = forests
    self.weighted = weighted
    self.print_results = print_results

    if self.kernel == 'adaptive':
        if self.print_results:
            print("Kernel: Adaptive\nNeightbours: {}".format(self.bw))

    elif self.kernel == 'fixed':
        if self.print_results:
            print("Kernel: Fixed\nBandwidth: {}".format(self.bw))

  def fit(self, X, y, c ):
    """
    Fit the model to the data.
    :param X: pandas DataFrame of shape (n_samples, n_features). Features data.
    :param y: pandas Series of shape (n_samples,). Target data.
    :param c: pandas DataFrame of shape (n_samples, 2). Coordinates data. (Longitude and Latitude)
    """



    self.coords = c
    self.X = X
    self.y = y

    self.Obs = len(self.X)

    if self.mtry is None:
        self.mtry = max(int(len(self.X)/3), 1)

    if self.forests:
        self.Gl_Model = RandomForestRegressor(n_estimators=self.ntree, max_features=self.mtry, n_jobs=-1)
        self.Gl_Model.fit(self.X.values, self.y.values)

        self.yhat = self.Gl_Model.predict(self.X.values)


    DistanceT = pdist(self.coords)
    Dij = squareform(DistanceT)

    if self.forests:
        self.lm_forests = [None] * self.Obs

    self.lm_LEst = pd.DataFrame({var: [] for var in self.X.columns})

    for m in range(self.Obs):
        # Get the data
        self.d_neighbour = Dij[:,m]
        self.dataset = pd.concat([self.X, self.y, pd.Series(self.d_neighbour, name='DNeighbour')], axis=1) # moze byc glad
        # Sort by distance
        self.dataset_sorted = self.dataset.sort_values(by='DNeighbour')
        if self.kernel == 'adaptive':
            # Keep Nearest Neighbours
            self.subset = self.dataset_sorted.iloc[:self.bw,:]
            self.kernel_h = self.subset['DNeighbour'].max()
        elif self.kernel == 'fixed':
            self.subset = self.dataset_sorted.loc[self.dataset_sorted['DNeighbour'] <= self.bw]
            self.kernel_h = self.bw
        # Bi-square weights
        self.wts = (1 - (self.subset['DNeighbour'] / self.kernel_h) ** 2) ** 2
        if self.weighted:
            self.local_model = RandomForestRegressor(n_estimators=self.ntree, max_features=self.mtry, 
                                                n_jobs=self.nthreads,oob_score=True)
            self.local_model.fit(self.subset[self.X.columns].values, self.subset[self.y.name].values, sample_weight=self.wts)
            self.local_predicted_y = self.local_model.predict(self.subset[self.X.columns].values)
            self.counter = 1
            while np.isnan(self.local_predicted_y).any():
                self.local_model = RandomForestRegressor(n_estimators=self.ntree, max_features=self.mtry, 
                                                    n_jobs=self.nthreads,oob_score=True)
                self.local_model.fit(self.subset[self.X.columns].values, self.subset[self.y.name].values, sample_weight=self.wts)
                self.local_predicted_y = self.local_model.predict(self.subset[self.X.columns].values)
                self.counter += 1
        else:
            self.local_model = RandomForestRegressor(n_estimators=self.ntree, max_features=self.mtry, 
                                                n_jobs=self.nthreads,oob_score=True)
            self.local_model.fit(self.subset[self.X.columns].values, self.subset[self.y.name].values)
            self.counter = 1
        if self.forests:
            self.lm_forests[m] = self.local_model
        # Importance
        self.lm_LEst.loc[m] = self.local_model.feature_importances_


  def predict(self,X,c,local_w=1,global_w=0 ):

    """
    Predicts the target variable using the trained random forest regression models.
    
    Args:
        X (pandas.DataFrame): The input feature data to be used for prediction.
        c (pandas.DataFrame): The coordinates of each observation in X.
        local_w (float, optional): Weight assigned to the local prediction. Defaults to 1.
        global_w (float, optional): Weight assigned to the global prediction. Defaults to 0.
        
    Returns:
        numpy.ndarray: Predicted values of the target variable.
    """




    predictions = np.zeros(len(X))
    for i in range(len(X)):

        D = np.sqrt((c[c.columns[0]][i] - self.coords[self.coords.columns[0]])**2 + (c[c.columns[1]][i] - self.coords[self.coords.columns[1]])**2)
        local_model_id = np.argmin(D)

        g_predict = self.Gl_Model.predict(X.iloc[i].values.reshape(1,-1))
        g_prediction = g_predict[0]

        l_predict = self.lm_forests[local_model_id].predict(X.iloc[i].values.reshape(1,-1))
        l_prediction = l_predict[0]

        predictions[i] = global_w * g_prediction + local_w * l_prediction
    
    return predictions

class GW_RandomForestClassifier():
  def __init__(self,
               kernel = 'adaptive',
               bw=100,
               ntree = 500,
               mtry = None,
               importance = 'impurity',
               nthreads = None,
               weighted = True,
               print_results = True,
               forests = True
               ):
    """
    Constructor for GW_RandomForestClassifier class.
    :param kernel: str, default 'adaptive'. Type of kernel to use for nearest neighbor selection. Options are 'adaptive' or 'fixed'.
    :param bw: int, default 100. Bandwidth for kernel selection.
    :param ntree: int, default 500. Number of trees to use for random forest.
    :param mtry: int, default None. Number of variables to consider for each split. If None, set to max(int(len(X)/3), 1).
    :param importance: str, default 'impurity'. Type of feature importance measure. Options are 'impurity' or 'permutation'.
    :param nthreads: int, default None. Number of threads to use for random forest.
    :param weighted: bool, default True. Whether to use weighted random forest.
    :param print_results: bool, default True. Whether to print results of kernel selection.
    :param forests: bool, default True. Whether to use random forests in local models.
    """

    self.bw = bw
    self.kernel = kernel
    self.ntree = ntree
    self.mtry = mtry
    self.importance = importance
    self.nthreads = nthreads
    self.forests = forests
    self.weighted = weighted
    self.print_results = print_results

    if self.kernel == 'adaptive':
        if self.print_results:
            print("Kernel: Adaptive\nNeightbours: {}".format(self.bw))

    elif self.kernel == 'fixed':
        if self.print_results:
            print("Kernel: Fixed\nBandwidth: {}".format(self.bw))

  def fit(self, X, y, c ):
    """
    Fit the model to the data.
    :param X: pandas DataFrame of shape (n_samples, n_features). Features data.
    :param y: pandas Series of shape (n_samples,). Target data.
    :param c: pandas DataFrame of shape (n_samples, 2). Coordinates data. (Longitude and Latitude)
    """



    self.coords = c
    self.X = X
    self.y = y

    self.Obs = len(self.X)

    if self.mtry is None:
        self.mtry = max(int(len(self.X)/3), 1)

    if self.forests:
        self.Gl_Model = RandomForestClassifier(n_estimators=self.ntree, max_features=self.mtry, n_jobs=-1)
        self.Gl_Model.fit(self.X.values, self.y.values)

        self.yhat = self.Gl_Model.predict(self.X.values)


    DistanceT = pdist(self.coords)
    Dij = squareform(DistanceT)

    if self.forests:
        self.lm_forests = [None] * self.Obs

    self.lm_LEst = pd.DataFrame({var: [] for var in self.X.columns})

    for m in range(self.Obs):
        # Get the data
        self.d_neighbour = Dij[:,m]
        self.dataset = pd.concat([self.X, self.y, pd.Series(self.d_neighbour, name='DNeighbour')], axis=1) # moze byc glad
        # Sort by distance
        self.dataset_sorted = self.dataset.sort_values(by='DNeighbour')
        if self.kernel == 'adaptive':
            # Keep Nearest Neighbours
            self.subset = self.dataset_sorted.iloc[:self.bw,:]
            self.kernel_h = self.subset['DNeighbour'].max()
        elif self.kernel == 'fixed':
            self.subset = self.dataset_sorted.loc[self.dataset_sorted['DNeighbour'] <= self.bw]
            self.kernel_h = self.bw
        # Bi-square weights
        self.wts = (1 - (self.subset['DNeighbour'] / self.kernel_h) ** 2) ** 2
        if self.weighted:
            self.local_model = RandomForestClassifier(n_estimators=self.ntree, max_features=self.mtry, 
                                                n_jobs=self.nthreads,oob_score=True)
            self.local_model.fit(self.subset[self.X.columns].values, self.subset[self.y.name].values, sample_weight=self.wts)
            self.local_predicted_y = self.local_model.predict(self.subset[self.X.columns].values)
            self.counter = 1
            while np.isnan(self.local_predicted_y).any():
                self.local_model = RandomForestClassifier(n_estimators=self.ntree, max_features=self.mtry, 
                                                    n_jobs=self.nthreads,oob_score=True)
                self.local_model.fit(self.subset[self.X.columns].values, self.subset[self.y.name].values, sample_weight=self.wts)
                self.local_predicted_y = self.local_model.predict(self.subset[self.X.columns].values)
                self.counter += 1
        else:
            self.local_model = RandomForestClassifier(n_estimators=self.ntree, max_features=self.mtry, 
                                                n_jobs=self.nthreads,oob_score=True)
            self.local_model.fit(self.subset[self.X.columns].values, self.subset[self.y.name].values)
            self.counter = 1
        if self.forests:
            self.lm_forests[m] = self.local_model
        # Importance
        self.lm_LEst.loc[m] = self.local_model.feature_importances_


  def predict(self,X,c,local_w=1,global_w=0 ):

    """
    Predicts the target variable using the trained random forest classification models.
    
    Args:
        X (pandas.DataFrame): The input feature data to be used for prediction.
        c (pandas.DataFrame): The coordinates of each observation in X.
        local_w (float, optional): Weight assigned to the local prediction. Defaults to 1.
        global_w (float, optional): Weight assigned to the global prediction. Defaults to 0.
        
    Returns:
        numpy.ndarray: Predicted values of the target variable.
    """




    predictions = np.zeros(len(X))
    for i in range(len(X)):

        D = np.sqrt((c[c.columns[0]][i] - self.coords[self.coords.columns[0]])**2 + (c[c.columns[1]][i] - self.coords[self.coords.columns[1]])**2)
        local_model_id = np.argmin(D)

        g_predict = self.Gl_Model.predict(X.iloc[i].values.reshape(1,-1))
        g_prediction = g_predict[0]

        l_predict = self.lm_forests[local_model_id].predict(X.iloc[i].values.reshape(1,-1))
        l_prediction = l_predict[0]

        predictions[i] = int(global_w * g_prediction + local_w * l_prediction)
    
    return predictions

