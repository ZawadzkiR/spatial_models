## Class GW_RandomForestRegressor
The GW_RandomForestRegressor class is a regression model based on a combination of global and local random forest models. Performs global regression on the entire data and local regression on each point using a weighted bi-square kernel to select neighboring points. It uses scikit-learn RandomForestRegressor to build global and local models. The code was rewritten from R to python based on the SpatialML library in R, which has the function of the Geographically Weighted Random Forest model. https://github.com/cran/SpatialML

## Class GW_RandomForestClassifier
The GW_RandomForestClassifier class is a classification model based on a combination of global and local random forest models. Performs global classification on the entire data and local classification on each point using a weighted bi-square kernel to select neighboring points. It uses scikit-learn RandomForestClassifier to build global and local models.


## Parameters
**kernel**: string (default: adaptive) - Type of kernel to be used. Either 'adaptive' or 'fixed'.
**bw**: float or int (default: 100) - Bandwidth for the kernel. If kernel='fixed', this value will be the maximum distance a point can be from the central point to be included in the local model. If kernel='adaptive', this value will be the number of neighbors included in the local model.
**ntree**: int (default: 500) - The number of trees in the random forest.
**mtry**: int (default: None) - The number of features to consider when looking for the best split in each tree. If None, it is set to the number of features divided by 3.
**importance**: string (default: impurity) - The method used to calculate the feature importance. Either 'impurity' or 'permutation'.
**nthreads**: int (default: None) - The number of threads used by the model when fitting the local models. If None, all available CPUs are used.
**weighted**: boolean (default: True) - Whether to use weights in the local model fitting.
**print_results**: boolean (default: True) - Whether to print the results of the model.
**forests**: boolean (default: True) - Whether to fit a global random forest model.
## Methods
**fit(X, y, c)**: Fits the model on the input data. X is the input feature matrix, y is the output target vector and c is the coordinates of each point.
**predict(X, c, local_w=1, global_w=0)**: Predicts the output for the input data. X is the input feature matrix and c is the coordinates of each point. local_w and global_w are the weights for the local and global models, respectively.
## Attributes
**bw**: float or int - The bandwidth used in the kernel.
**kernel**: string - The type of kernel used.
**ntree**: int - The number of trees in the random forest.
**mtry**: int - The number of features to consider in the split.
**importance**: string - The method used to calculate feature importance.
**nthreads**: int - The number of threads used in the local model fitting.
**weighted**: boolean - Whether the local model fitting is weighted.+
**print_results**: boolean - Whether the results are printed.
**forests**: boolean - Whether a global random forest model is fitted.
**coords**: dataframe - The coordinates of each point.
**X**: dataframe - The input feature matrix.
**y**: dataframe - The output target vector.
**Obs**: int - The number of observations in the input data.
**Gl_Model**: object - The global random forest model.
**yhat**: array - The predicted output of the global model.
**lm_forests**: array - The local random forest models.
**lm_LEst**: dataframe - The feature importance of each local model.


