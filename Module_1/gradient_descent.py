#%% imports
import numpy as np
import matplotlib.pyplot as plt

#%% definitions
# primitive gradient descent
def minimize(function, initial_guess, learning_rate = 0.01, tolerance=1E-6):
    current_step = initial_guess
    current_value = function(current_step)
    delta = tolerance + 1
    while(abs(delta) > tolerance):
        derivative = differentiate(function, current_step)
        next_step = current_step - learning_rate * current_value * derivative / (np.linalg.norm(derivative))**2
        delta = function(next_step) - current_value
        current_step = next_step
        current_value += delta
    
    return current_step

def differentiate(function, variables, delta=1E-10):
    functional_value = function(variables)
    num_variables = len(variables)
    result = np.zeros(num_variables)
    for i in range(num_variables):
        d_variable = np.zeros(num_variables)
        d_variable[i] = delta
        result[i] = (function(variables + d_variable) - function(variables - d_variable)) / (2 * delta)
        if abs(functional_value / result[i]) < delta:
            print('function is changing too fast')

    return result

def least_squares_regression(fit_function, data_x, data_y, initial_guess):
    function_to_minimize = lambda parameters : least_squares_error_function(fit_function, parameters, data_x, data_y)
    best_fit_parameters = minimize(function_to_minimize, initial_guess)
    return best_fit_parameters

def least_squares_error_function(fit_function, parameters, x, y):
    dataset_size = len(x)
    return (1 / dataset_size) * np.sum((fit_function(x, parameters) - y)**2)

#%% test code
x = np.linspace(0,1,20)
y = 2.87 * x + 3.487 + np.random.normal(loc = 0, scale = 0.1, size = x.shape)

def linear_fit(x, params):
    a, b = params[0], params[1]
    return a * x + b

popt = least_squares_regression(linear_fit, x, y, np.array([0,0]))
print(popt)

plt.plot(x, linear_fit(x, popt))
plt.scatter(x, y)
plt.show()

# NOTES
'''
In the future it would be great to add the option to use analytic gradient rather than numerical
The way the gradient descent is currently written up might be flawed for cases where the minimized least squares is still non zero and large -- need to revisit this
'''
