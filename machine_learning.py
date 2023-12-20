
import pandas as pd
import numpy as np

######################## preprocess_classification_dataset STARTS ######################## 
def preprocess_classification_dataset():
    
    ########## train dataset ##########
    train_df = pd.read_csv('train.csv')
    
    # To get features columns from test_df except the output column (last column)
    train_feat_df = train_df.iloc[:,:-1] 
    train_output = train_df[['output']]
    
    # To Convert the feature and output dataframes into numpy arrays
    X_train = train_feat_df.values
    y_train = train_output.values
    
    
    ########## val dataset ##########
    val_df = pd.read_csv('val.csv')
    
    val_feat_df = val_df.iloc[:,:-1]
    val_output = val_df[['output']]
    
    X_val = val_feat_df.values
    y_val = val_output.values
    
    
    ########## test dataset ##########
    test_df = pd.read_csv('test.csv')
    
    test_feat_df = test_df.iloc[:,:-1]
    test_output = test_df[['output']]
    
    X_test = test_feat_df.values
    y_test = test_output.values
    
    return X_train, y_train, X_val, y_val, X_test, y_test 
######################## preprocess_classification_dataset ENDS ########################  


 
  
######################## knn_classification STARTS ######################## 
def knn_classification(X_train, y_train, x_new , k = 5):
    
    pair_value = []
    
    for y in range(len(y_train)):
        dist = 0
        dist += euclideanDistance(x_new, X_train[y], len(X_train[0]))
            
        pair_value.append((dist, y_train[y][0]))
    
    sorted_dist = sorted(pair_value, key = lambda n: n[0])
    K_sorted_list = sorted_dist[:k]
    
    y_value = []
    for i in range(len(K_sorted_list)):
        y_value.append(K_sorted_list[i][1])
    
    y_ferq = np.unique(y_value, return_counts = True)
    
    if y_ferq[1][0] >= k / 2:
        if y_ferq[0][0] == 0:
            return 0
          
        else:
            return 1
    else:
        return 1
######################## knn_classification ENDS ######################## 
        



def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance2[x] - instance1[x]), 2)
    return np.sqrt(distance)



######################## logistic_regression_training STARTS ######################## 
def logistic_regression_training(X_train, y_train, alpha = 0.01, max_iters = 5000, random_seed = 1):
    
    num_of_features = np.hstack((np.ones([X_train.shape[0], 1], dtype = int), X_train))
    np.random.seed(random_seed) # for reproducibility
    
    weights = np.random.normal(loc=0.0, scale=1.0, size=(len(num_of_features[0]), 1)) # size = lenght
    
    while max_iters > 0:
        
        maxtrix = num_of_features @ weights
        logistic_line = 1 / (1 + np.e**(-maxtrix))
        
        temp_error = logistic_line - y_train
        error = alpha * num_of_features.T @ temp_error
        
        weights = weights - error
        max_iters = max_iters - 1
    
    return weights
######################## logistic_regression_training ENDS ######################## 




######################## logistic_regression_prediction STARTS ######################## 
def logistic_regression_prediction(X, weights, threshold = 0.5):
    
    num_of_features = np.hstack((np.ones([X.shape[0], 1], dtype = int), X))
    maxtrix = num_of_features @ weights
    logistic_line = 1 / (1 + np.e**(-maxtrix))
    
    y_preds = []
    y_preds =  logistic_line
    
    for i in range(len(logistic_line)):
        
        if logistic_line[i] >= threshold:
            y_preds[i] = 1
            
        else:
            y_preds[i] = 0
            
    return y_preds
######################## logistic_regression_prediction ENDS ######################## 



######################## model_selection_and_evaluation STARTS ######################## 
def model_selection_and_evaluation(alpha = 0.01, max_iters = 5000, random_seed = 1, threshold = 0.5):
    
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_classification_dataset()
    one = [] 
    three = []
    five = []
    
    for i in range(len(X_val)):
        one.append(knn_classification(X_train, y_train, X_val[i], 1))
        three.append(knn_classification(X_train, y_train, X_val[i], 3))
        five.append(knn_classification(X_train, y_train, X_val[i], 5))
        
    Logistic = logistic_regression_prediction(X_val, logistic_regression_training(X_train, y_train, alpha, max_iters, random_seed), threshold)
    val_accuracy_list =[]
    
    val_accuracy_list.insert(0,(y_val.flatten() == np.array(one).flatten()).sum() / y_val.shape[0])
    val_accuracy_list.insert(1,(y_val.flatten() == np.array(three).flatten()).sum() / y_val.shape[0])
    val_accuracy_list.insert(2,(y_val.flatten() == np.array(five).flatten()).sum() / y_val.shape[0])
    
    val_accuracy_list.insert(3,(y_val.flatten() == Logistic.flatten()).sum() / y_val.shape[0])
    models = ['1nn', '3nn', '5nn', 'logistic regression']
    position = val_accuracy_list.index(sorted(val_accuracy_list)[-1])
    best_method = models[position]
    X_train_val_merge = np.vstack([X_train, X_val]) 
    y_train_val_merge = np.vstack([y_train, y_val])
    y_test_pred = []
    if position != 3:
        for i in range(len(X_test)):
            y_test_pred.append(knn_classification(X_train_val_merge, y_train_val_merge, X_test[i], 2 * position + 1))
            
        test_accuracy = sum(np.array(y_test.flatten() == np.array(y_test_pred).flatten())) / y_test.shape[0]
        
    else:
        y_test_pred = logistic_regression_prediction(X_test, logistic_regression_training(X_train_val_merge, y_train_val_merge, alpha, max_iters, random_seed), threshold)
        test_accuracy = sum(np.array(y_test.flatten() == np.array(y_test_pred).flatten())) / y_test.shape[0]
        
    return best_method, val_accuracy_list, test_accuracy
######################## model_selection_and_evaluation ENDS ######################## 
