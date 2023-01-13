import KernelRidge from sklearn
import mean_absolute_error from sklearn.metrics
import r2_score from sklearn.metrics
import matplotlib as plt
import train_test_split from sklearn.model_selection
import numpy as np


def krr_train(X_train, y_train, X_test, y_test, alphas, gammas):
    '''
    Function to train krr models
    Input args:
        X_train: Training feature set
        y_train: Training predictor set
        X_test: Testing feature set
        y_test: Testing predictor set
    Returns -> best_model, best_params
    '''
    
    # Setup
    info, mae = [], []
    
    #Search grid
    for alpha in alphas:
        for gamma in gammas:
            print("Current params: alpha = " + alpha + ", gamma = " + gamma)
            model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
            model.fit(X_train, y_train)
            error = mean_absolute_error(y_test, model.predict(X_test))
            mae.append(error)
            info.append((alpha, gamma, error))
    
    # Best parameters
    best_params = info[mae.index(min(mae))]
    best_model = KernelRidge(kernel='rbf', alpha=best_params[0], gamma=best_params[1])
    best_model.fit(X_train, y_train)

    return best_model, best_params


def krr_evaluate(model, X_test, y_test):
    '''
    Function to evaluate krr models
    Input args:
        model: KRR model with params
        X_test: Testing feature set
        y_test: Testing predictor set
    Returns -> r2_value, mae_value
    '''
    # Evaluation metrics
    r2_value = r2_score(model.predict(X_test), y_test)
    mae_value = mean_absolute_error(model.predict(X_test), y_test)
    
    print("R2 Score = ", r2_value)
    print("MAE = ", mae_value)

    
    plt.plot(y_test, model.predict(X_test), 'b.')
    plt.plot(parity, parity, 'black', linestyle='--', lw = '1')
    plt.grid()
    plt.title('ML forces (Predicted) vs. QM forces (Actual)')
    plt.xlabel("QM Forces (eV/A)")
    plt.ylabel("ML Forces (eV/A)")
    
    return r2_value, mae_value

    
def krr_run(X, y):
    '''
    Function to run krr training models
    Input args:
        X: Fingerprints
        y: Corresponsing forces
    Returns -> final_mae
    '''
    # Search space for hyperparameters
    alphas = np.logspace(-5, -5, 11)
    gammas = np.logspace(-5, -5, 11)
    
    # Splitting into training and testing part
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    
    
    # Train the model
    model, best_params = krr_train(X_train, y_train, X_test, y_test, alphas, gammas)
    print("Best Hyperparameters: ", best_params)


    mae_final = krr_evaluate(model, X_test, y_test)
    
    return final_mae, [X_train, X_test, y_train, y_test]
