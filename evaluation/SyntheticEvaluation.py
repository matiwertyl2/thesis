import numpy as np
from sklearn.linear_model import Lasso

TINY = 1e-12

def normalize(X, mean=None, stddev=None, useful_features=None, remove_constant=True):
    calc_mean, calc_stddev = False, False
    
    if mean is None:
        mean = np.mean(X, 0) # training set
        calc_mean = True
    
    if stddev is None:
        stddev = np.std(X, 0) # training set
        calc_stddev = True
        useful_features = np.nonzero(stddev)[0] # inconstant features, ([0]=shape correction)
    
    if remove_constant and useful_features is not None:
        X = X[:, useful_features]
        if calc_mean:
            mean = mean[useful_features]
        if calc_stddev:
            stddev = stddev[useful_features]
    
    X_zm = X - mean    
    X_zm_unit = X_zm / stddev
    
    return X_zm_unit, mean, stddev, useful_features

def mse(predicted, target):
    ''' mean square error '''
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted #(n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target #(n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0] #value not array

def rmse(predicted, target):
    ''' root mean square error '''
    return np.sqrt(mse(predicted, target))

def nmse(predicted, target):
    ''' normalized mean square error '''
    return mse(predicted, target) / np.var(target)

def nrmse(predicted, target):
    ''' normalized root mean square error '''
    return rmse(predicted, target) / np.std(target)


def norm_entropy(p):
    '''p: probabilities '''
    n = p.shape[0]
    return - p.dot(np.log(p + TINY) / np.log(n + TINY))

def entropic_scores(r):
    '''r: relative importances '''
    r = np.abs(r)
    ps = r / (np.sum(r, axis=0) + TINY) # 'probabilities'
    hs = [1-norm_entropy(p) for p in ps.T]
    return hs


def disentanglement_score(R):
    # R - matrix of relative importance 
    # returns : disentanglement score

    disent_scores = entropic_scores(R.T)
    c_rel_importance = np.sum(R,1) / np.sum(R) # relative importance of each code variable
    disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)
    return disent_w_avg

def completeness_score(R):
    # R - matrix of relative importance 
    # returns : completeness score
    complete_scores = entropic_scores(R)
    complete_avg = np.mean(complete_scores)
    return complete_avg

def informativeness_score(z_pred, z_real, error_function):
    # z_pred - predictions of generative factors (data_size x z)
    # z_real - real generative factors (data_size x z)
    
    errors = np.zeros(z_real.shape[1])
    for i in range(z_real.shape[1]):
        errors[i] = error_function(z_pred[:, i], z_real[:, i])
    
    return errors.mean()

def lasso_regressor(x, y):
    model = Lasso(alpha=0.02)
    model.fit(x, y)
    r = getattr(model, 'coef_')[:, None] # [n_c, 1]
    return model, np.abs(r)

def evaluate_model(random_factors, latent_repr, regressor=lasso_regressor, error_function=nrmse):
    # evaluates disentanglement, completeness and informativeness
    # params:
    # random_factors - data_size x z array of random_factors
    # latent_repr - latent representation of dataset samples, computed by the model that we 
    # want to evaluate
    # regressor  - predicts the generative factors based on latent repr, 
    # takes latent_repr and generative factors as arguments, 
    # returns n x z matrix of predictions of generative factors 
    # and latent_dim x z matrix of relative importance         
    # error_function - takes prediction and true value and computes some error value  

    n = random_factors.shape[0]
    z = random_factors.shape[1]
    train_size = int(n / 2)  # train/ test split in half
    test_size = n - train_size
    R =  []
    predictions = []
    #normalize input and output
    random_factors_train, mu, stddev, _ = normalize(random_factors[:train_size, :], remove_constant=False)
    random_factors_test = normalize(random_factors[train_size:, :], mean=mu, stddev=stddev, remove_constant=False)[0]
    random_factors = np.vstack((random_factors_train, random_factors_test))
    latent_repr = normalize(latent_repr)[0]

    for i in range(z):
        model, r = regressor(latent_repr[:train_size, :], random_factors[:train_size, i])
        predictions.append(model.predict(latent_repr[train_size:, :]).reshape(test_size, 1))
        R.append(r)

    R = np.hstack(R)
    predictions = np.hstack(predictions)

    disentanglement = disentanglement_score(R)
    completeness = completeness_score(R)
    informativeness = informativeness_score(predictions, random_factors[train_size:, :], error_function)
    return disentanglement, completeness, informativeness


__all__ = ['evaluate_model']