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
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height =fig_width*golden_mean       # height in inches
    fig_size = (fig_width,fig_height)
    return fig_size

def opL_mat(n):
    """Compute the operator L matrix in dim n
    INPUT :
        n = dimension of the matrix
    OUTPUT
        L_mat = operator L matrix
    """
    L_mat = -np.eye(n) + np.diag(np.ones(n-1), 1)
    L_mat[n-1, n-1] = 0
    return L_mat

def opL(x):
    """Apply operator L to vetctor x
    INPUT
        x = input vector onto which L is applied
    OUTPUT
        L(x)
    """
    return np.dot(opL_mat(x.size), x)

def opL_star(x):
    """Apply L* (adjoint operator of L) to vector x
    INPUT
        x = vector onto which L* is applied
    OUTPUT
        L*(x)
    """
    return np.dot(opL_mat(x.size).transpose(), x)

def prox_ic_factory(mu):
    """return a projection function on the convex {|u|inf <= mu / u in R^p}
    Generate a function to use whith the forwarf_backward_factory
    INPUT
        mu = parameter of the convex set
    OUTPUT
        P : R x R^k -> R^k
        P first argument is discarded
        P(gamma,u) = is the projection of u on the convex {|u|inf <= mu / u in R^p}
    """
    if mu == np.Infinity :
        return lambda gamma, u : u
    else :
        return lambda gamma, u : -mu * (u  < -mu) + u *((u >= -mu) & (u <= mu)) + mu * (u > mu)

def grad_g_factory(y):
    """Return the gradient function of u |-> ||L*u - y||^2
    INPUT
        y = parameter of the function
    OUTPUT
        grad_g : R^k -> R^k
        u |-> grad_g(u)
    """
    return lambda x : opL(opL_star(x) - y)

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

def apply_forward_backward(x0, algo, convert_sol, N = 1000, primal_objective_function = None, dual_objective_function = None):
    """apply the forward backward algorithm
    INPUT
        x0 = first iterate
        algo = the forward backaward step function
        convert_sol = function that convert the dual solution into the primal one
        N = iteration number
        primal_objective_function = the primal objective function, if given the function will return the value of this objective function for each iterate
        dual_objective_function = the dual objective function , if given the function wiil return the value of this objective function for each iterate
    OUTPUT
        x_N = result of the foward backward algorithm

        if one of the ojective function was given:
        (x_N, obj_primal,obj_dual) where obj_primal and obj_dual are None if the corresponding function was not specified and a N+1 vector if the corresponding function was given
    """
    x = x0.copy()
    if primal_objective_function != None:
        obj_primal = np.empty((N+1,))
        obj_primal[0] = primal_objective_function(convert_sol(x0))
    else:
        obj_primal == None
    if dual_objective_function != None:
        obj_dual = np.empty((N+1,))
        obj_dual[0] = dual_objective_function(x0)
    else:
        obj_dual = None

    for i in range(N):
        x = algo(i, x)
        if primal_objective_function != None:
            obj_primal[i+1] = primal_objective_function(convert_sol(x))
        if dual_objective_function != None:
            obj_dual[i+1] = dual_objective_function(x)
    x = convert_sol(x)
    if primal_objective_function != None and dual_objective_function != None:
        return (x, obj_primal, obj_dual)
    else:
        return x

def primal_obj_function(y, l):
    return lambda x :  0.5 * np.linalg.norm(x-y)**2 + l * np.linalg.norm(opL(x), 1)
def dual_obj_function(y, l):
    return lambda x : 0.5 * np.linalg.norm(y - opL_star(x)) ** 2

def TP_ex1_experiment(x, x_rand, l, gamma_coeff, N_iter, lambda_n, plot_fig = False):
    N = x.size
    t = np.arange(0, x.size, 1)
    gamma = gamma_coeff / (np.linalg.norm(np.dot(opL_mat(N), opL_mat(N).transpose())))
    u_0 = np.zeros(N)
    algo = forward_backward_factory(grad_g_factory(x_rand), prox_ic_factory(l), gamma, 1)
    convert_sol = lambda x : x_rand - opL_star(x)
    x_hat, obj_primal, obj_dual = apply_forward_backward(u_0,
        algo,
        convert_sol,
        N_iter,
        primal_objective_function = primal_obj_function(x_rand, l),
        dual_objective_function = dual_obj_function(x_rand, l))
    if plot_fig:
        fig = plot.figure(figsize = get_fig_params(236.01561))
        plot.plot(t, x, label='signal original')
        plot.plot(t, x_hat, label ='signal estimé')
        plot.xlabel('t')
        plot.ylabel('x')
        plot.legend()
        fig.savefig('figures/xhat_'+str(l)+'_'+str(gamma_coeff)+'_'+str(lambda_n)+'.pgf')
        fig.savefig('figures/xhat_'+str(l)+'_'+str(gamma_coeff)+'_'+str(lambda_n)+'.pdf')

        fig = plot.figure(figsize = get_fig_params(236.01561))
        plot.subplot(211)
        plot.plot(range(N_iter + 1), obj_primal, label='Objectif primal')
        plot.plot(range(N_iter + 1), obj_dual, label='Objectif dual')
        plot.xlabel('Iteration number')
        plot.ylabel('objective')
        plot.xscale('log')
        plot.legend()

        plot.subplot(212)
        plot.plot(range(N_iter +  1), obj_primal + obj_dual, label = 'duality gap')
        plot.xscale('log')
        plot.yscale('log')
        plot.legend()
        plot.xlabel('Iteration number')
        plot.ylabel('Duality gap')
        fig.savefig('figures/objectives_functions_'+str(l)+'_'+str(gamma_coeff)+'_'+str(lambda_n)+'.pgf')
        fig.savefig('figures/objectives_functions_'+str(l)+'_'+str(gamma_coeff)+'_'+str(lambda_n)+'.pdf')

    return x_hat

def TP_ex1_experiment_mse(x, x_rand, l, gamma_coeff,  N_iter, lambda_n, plot_fig = False):
    N = x.size
    gamma = gamma_coeff / (np.linalg.norm(np.dot(opL_mat(N), opL_mat(N).transpose())))
    u_0 = np.zeros(N)
    convert_sol = lambda x : x_rand - opL_star(x)
    mse = np.empty((len(l),))
    c = 0
    for current_l in l:
        algo = forward_backward_factory(grad_g_factory(x_rand), prox_ic_factory(current_l), gamma, 1)
        x_hat, obj_primal, obj_dual = apply_forward_backward(u_0,
            algo,
            convert_sol,
            N_iter,
            primal_objective_function = primal_obj_function(x_rand, current_l),
            dual_objective_function = dual_obj_function(x_rand, current_l))
        mse[c] = np.linalg.norm(x - x_hat)
        c += 1

    if plot_fig:
        fig = plot.figure(figsize = get_fig_params(236.01561))
        plot.plot(l ,mse, label='Mean square error')
        plot.xscale('log')
        plot.xlabel('$\\lambda$')
        plot.ylabel('error')
        plot.legend()
        fig.savefig('figures/mse_'+str(gamma_coeff)+'_'+str(lambda_n)+'.pgf')
        fig.savefig('figures/mse_'+str(gamma_coeff)+'_'+str(lambda_n)+'.pdf')

    return mse

#load data for Excercice 1
data_ex1 = sio.loadmat("Exercie1/signal_ex1.mat")
x = data_ex1['x'][0]
x_rand = x + np.random.randn(x.size)


#Fig 1 : Signals
t = np.arange(0, x.size)
fig = plot.figure(figsize = get_fig_params(236.01561))
plot.plot(t, x, label='signal original')
plot.plot(t, x_rand, label ='signal bruité')
plot.xlabel('t')
plot.ylabel('x')
plot.legend()
plot.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
fig.savefig('figures/signaux.pgf')
fig.savefig('figures/signaux.pdf')


temp = TP_ex1_experiment(x, x_rand, 0.1, 1, 9000, 0.9, True)
temp = TP_ex1_experiment(x, x_rand, 1, 1, 6000, 0.9, True)

TP_ex1_experiment(x, x_rand, 1.5, 1, 6000, 0.9, True)
TP_ex1_experiment(x, x_rand, 2, 1.5, 6000, 0.6, True)
TP_ex1_experiment(x, x_rand, 2, 1, 6000, 0.9, True)
TP_ex1_experiment(x, x_rand, 2, 1.5, 6000, 0.1, True)
TP_ex1_experiment(x, x_rand,5, 1, 6000, 0.9, True)
TP_ex1_experiment(x, x_rand, 100, 1, 10000, 0.9, True)

# lambda_list = [0, 0.1,0.5]
lambda_list = [0, 0.1,0.5, 0.7, 1,2,3,4, 5, 10, 100]
TP_ex1_experiment_mse(x, x_rand, lambda_list, 1, 10000, 1, True)
plot.close('all')
