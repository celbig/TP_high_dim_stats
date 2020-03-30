import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plot
import matplotlib
matplotlib.use("pgf")
plot.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'figure.autolayout' : True,
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'lines.linewidth':1,
})

#determine fig size
def get_fig_params(fig_width_pt):
    """compute the desired ratio for a figure in inches
    INPUT:
        fig_width_pt : desired width in latex pt
    OUTPUT:
        fig_width : witdh of the figure in inches
        fig_height : height of the figure in inches
    """
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height =fig_width*golden_mean       # height in inches
    fig_size = (fig_width,fig_height)
    return fig_size

#load data for Excercice 1
data_ex2 = sio.loadmat("Exercice2/data_exo2_training.mat")
b = data_ex2['b']
y = data_ex2['y']
N = y.shape[0]
K = y.shape[1]

data_est_ex2 = sio.loadmat("Exercice2/data_exo2_estimation.mat")
b_est = data_est_ex2['b']
y_est = data_est_ex2['y']
N_est = y_est.shape[0]

print('Number of patient in trainig dataset :')
print(b.size)
print('Number of healthy patient')
print(np.sum(b>0))
print('Number of sick patient')
print(np.sum(b<0))

print('Number of patient in test dataset :')
print(b_est.size)
print('Number of healthy patient')
print(np.sum(b_est>0))
print('Number of sick patient')
print(np.sum(b_est<0))

def primal_obj_function_factory(y, b, l):
    """return the objective function based on dataset Parameters
    returns a function
    INPUT:
        y in R^NxK : predictives variables for N patients
        b in R^k : state of each patient -> 1 healthy, -1 ill
        l : regularisation parameter lambda
    OUTPUT :
        f in R^k -> R : the objective function
    """
    def return_function(x):
        return l * np.linalg.norm(x, 1) + np.sum(np.log1p(np.exp(-b * np.dot(x.reshape((1, x.size)), y.transpose()))))
    return return_function

def predict(x, y):
    """Predict the state for a set of patient y based on estimator x
    INPUTS :
        x in R^K : estimator
        y in R^NxK : predictives variables for N patients
    OUTPUTS :
        b in R^k : state of each patient : 1 healthy, -1 ill
    """

    return np.sign(np.dot(x.transpose(), y.transpose()))

def calc_nu(y):
    """Compute grad f Lipschitz-constant ν
    INPUT:
        y : y in R^NxK : predictives variables for N patients
    OUTPUT:
        nu : a majoration of ν
    """
    K = b.size
    y_abs = np.abs(y)
    M = np.dot(y_abs.transpose(), y_abs)
    return np.linalg.norm(M)

def prox_L1_factory(l):
    """Generate the prox operator of gamma * lambda * ||.||1
    Returns a function
    INPUT :
        l : lambda
    OUTPUT:
        prox_gamma_lambda_L1 in RxR^K -> R : prox_gamma*lambda*f (x)
        """
    return lambda gamma, x : (x-(gamma*l)) * (x > (gamma*l)) +  (x+(gamma*l)) * (x < -(gamma*l))

def grad_g_factory(y, b):
    """Generate the gradient of g based on Parameters
    Returns a function
    INPUT:
        y in R^NxK : predictives variables for N patients
        b in R^k : state of each patient -> 1 healthy, -1 il
    OUTPUT:
        grad_g i R^K -> R^k : the gradient function of g
        """

    def return_function(x):
        arg =  np.repeat(b * np.dot(x.reshape((1, x.size)), y.transpose()), y.shape[1], axis = 0)
        return np.sum((np.repeat(-b, y.shape[1], axis = 0) * y.transpose()) * 1/(np.exp(arg)+1), axis = 1)
    return return_function

def forward_backward_factory(grad_g, prox_gamma_f, gamma, lambda_n):
    """return the next_step function for the forward backward algorithm
    INPUT
        grad_g = R^k -> R^k the gradient function of g
        prox_gamma_f = R x R^k -> R^k the prox function of gamma f
            s.t. prox_gamma_f(gamma, u) = prox_gamma*f(u)
        gamma = the gamma parameter of the forward backward algorithm
        lambda_n = either a constant or a function N -> R
    OUTPUT
        forward_backward : the step function of the forward backward algo
        N x R^k -> R^k
        forward_baward(n, x_n) = x_n+1
    """

    if not callable(lambda_n):
        flambda_n = lambda n : lambda_n
    else:
        flambda_n = lambda_n
    return lambda n, x : x + flambda_n(n) * (prox_gamma_f(gamma, x - gamma * grad_g(x)) - x)

def apply_forward_backward(x0, algo, N = 1000, primal_objective_function = None):
    """apply the forward backward algorithm
    INPUT
        x0 = first iterate
        algo = the forward backaward step function
        N = iteration number
        primal_objective_function = the primal objective function, if given the function will return the value of this objective function for each iterate
    OUTPUT
        x_N = result of the foward backward algorithm
        OR (x_N, obj_primal) if the ojective function was given, with obj_primal the vector of primal objectives for each iterate
    """
    x = x0.copy()
    if primal_objective_function != None:
        obj_primal = np.empty((N+1,))
        obj_primal[0] = primal_objective_function(x0)
    else:
        obj_primal = None


    for i in range(N):
        x = algo(i, x)
        if primal_objective_function != None:
            obj_primal[i+1] = primal_objective_function(x)

    x = x.reshape((x.size,1))
    if primal_objective_function != None :
        return (x, obj_primal)
    else:
        return x


#Parameters
N_iter =  15000
l_list = [0.1, 1, 2,  10]
grad = grad_g_factory(y, b)
nu = calc_nu(y)
gamma = 1/ nu
print("gamma : "+ str(gamma))


for l in l_list:
    print('lambda = ' + str(l))
    prox_L1 = prox_L1_factory(l)
    primal_obj_fun = primal_obj_function_factory(y, b, l)

    #Compute the estimator
    algo = forward_backward_factory(grad, prox_L1, gamma, 1.25)
    x_hat, primal_obj = apply_forward_backward(np.zeros((1, K)), algo, N_iter, primal_obj_fun)
    sparcity = np.sum(1.0*(x_hat==0)) / x_hat.size
    print('sparcity :' + str(sparcity))
    # Estimate accuracy on training set
    print('accuracy on training set :')
    print(np.sum(1 * (b[0]  ==  predict(x_hat, y)[0])) /b.size)

    #plot the objective function
    fig = plot.figure(figsize = get_fig_params(236.01561))
    plot.plot(np.arange(N_iter+1), primal_obj, label = 'Objectif primal')
    plot.xlabel('Nombre d\'itérations')
    plot.ylabel('Objectif')
    plot.xscale('log')
    plot.legend()
    fig.savefig('figures/ex2_objective_'+str(l)+'.pgf')
    fig.savefig('figures/ex2_objective_'+str(l)+'.pdf')
    plot.close('all')

    # Estimate accuracy on testing set
    print('accuracy on testing set :')
    print(np.sum(b_est[0] == predict(x_hat, y_est)[0]) / b_est.size)
    print('\n\n')


#test invalid gamma
l = 2
gamma = 10**(-4)
print("gamma : "+ str(gamma))
print('lambda = ' + str(l))
prox_L1 = prox_L1_factory(l)
primal_obj_fun = primal_obj_function_factory(y, b, l)

#Compute the estimator
algo = forward_backward_factory(grad, prox_L1, gamma, 1.25)
x_hat, primal_obj = apply_forward_backward(np.zeros((1, K)), algo, N_iter, primal_obj_fun)

# Estimate accuracy on training set
print('accuracy on training set :')
print(np.sum(1 * (b[0]  ==  predict(x_hat, y)[0])) /b.size)

#plot the objective function
fig = plot.figure(figsize = get_fig_params(236.01561))
plot.plot(np.arange(N_iter+1), primal_obj, label = 'Objectif primal')
plot.xlabel('Nombre d\'itérations')
plot.ylabel('Objectif')
plot.xscale('log')
plot.legend()
fig.savefig('figures/ex2_non_convergent_objective_'+str(l)+'.pgf')
fig.savefig('figures/ex2_non_convergent_objective_'+str(l)+'.pdf')
print('accuracy on testing set :')
print(np.sum(b_est[0] == predict(x_hat, y_est)[0]) / b_est.size)
