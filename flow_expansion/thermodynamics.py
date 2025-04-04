import torch
import numpy as np
from scipy.stats import gaussian_kde
from typing import List, Tuple
import matplotlib.pyplot as plt

#Temperature dependant partition function via flow
def lnZt_func_inn(inn, T: np.ndarray, DEVICE, N_DIM: int, nsamples: int = 125000, plot: bool = False, save: bool = False, path_to_save: str = None, comparison: bool = True, samples: np.ndarray = None) -> np.ndarray:
    alphas = torch.randn(nsamples, N_DIM).to(DEVICE).double()
    thetas, det_theta_alpha = inn(alphas, rev=True)
    alphas, thetas, det_theta_alpha= alphas.cpu().detach().numpy(), thetas.cpu().detach().numpy(), det_theta_alpha.cpu().detach().numpy()
    
    #Precomputations
    alpha_sum = np.sum(alphas**2, axis=1)
    exp_alpha = np.exp(-1/2 * alpha_sum)
    
    exp_alpha_term = (exp_alpha[:,None]**(1/T-1).T)
    exp_det_term = np.exp(det_theta_alpha)[:,None]**(1-1/T).T
    
    log_sum = np.log((exp_alpha_term * exp_det_term).sum(axis = 0))

    #Evidence from Multinest
    lnpy = - 241.08 + 356.79
    
    lnZt = lnpy / T - np.log(nsamples) + (1 -1/T) * np.log(2*np.pi) + log_sum

    if plot:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        plt.rcParams.update({'font.size': 16})
        
        plt.semilogx()
        plt.plot(T,lnZt)
        plt.xlabel('$T$')
        plt.ylabel('ln $Z[T]$')
        plt.vlines(x=1, ymin=np.min(lnZt), ymax=np.max(lnZt), color='black', linestyle='--', label = '$T = 1$ (evidence)')
        #plt.vlines(x=1, ymin=-2400, ymax=0, color='black', linestyle='--', label = '$T = 1$ (evidence)')
        plt.legend(loc = 'upper right')
        if save:
            plt.savefig(path_to_save+'lnZt.pdf', dpi = 300, bbox_inches='tight')
        plt.show()
        
    if comparison:
        lnZt_classic, _, _ = lnZt_from_posterior_estimate(samples, T)
        if plot:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
            plt.rcParams.update({'font.size': 16})
            
            plt.semilogx()
            plt.plot(T,lnZt, label = 'INN', color = 'tab:blue')
            plt.plot(T,lnZt_classic, label = 'KDE', color = 'tab:orange', linestyle = ':')
            plt.xlabel('$T$')
            plt.ylabel('ln $Z[T]$')
            plt.vlines(x=1, ymin=np.min(lnZt), ymax=np.max(lnZt), color='black', linestyle='--', label = '$T = 1$ (evidence)')
            #plt.vlines(x=1, ymin=-2400, ymax=0, color='black', linestyle='--', label = '$T = 1$ (evidence)')
            plt.legend(loc = 'upper right')
            if save:
                plt.savefig(path_to_save+'lnZt_flow_kde.pdf', dpi = 300, bbox_inches='tight')
            plt.show()
    return lnZt

#Temperature dependant partition function via KDE
def lnZt_from_posterior_estimate(samples: np.ndarray, T: np.ndarray, grid_size: int = 100, plot: bool = False, save: bool = False, path_to_save: str = None, with_out_evidence: bool = False) -> float:
    #KDE estimate of posterior
    def posterior_kde_estimate(samples: np.ndarray, grid_size: int = 100):
        kde = gaussian_kde(samples.T)
        
        # Create a grid to evaluate the KDE over the range of the samples
        x_min, y_min = samples.min(axis=0)
        x_max, y_max = samples.max(axis=0)
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()])
        
        # Evaluate the KDE on the grid
        pdf_values = kde(grid)
        
        # Reshape the evaluated pdf values back into the grid shape
        pdf_values = pdf_values.reshape(X.shape)
        
        # Ensure the probability density values sum to one (discrete approximation of integral)
        dx = (x_max - x_min) / (grid_size - 1)
        dy = (y_max - y_min) / (grid_size - 1)
        pdf_values /= np.sum(pdf_values) * dx * dy  # Normalizing PDF over the grid
        return pdf_values, dx, dy

    pdf_values, dx, dy = posterior_kde_estimate(samples, grid_size)
    
    # Calculate the entropy using the estimated PDF
    pdf_nonzero = pdf_values[pdf_values > 0]  # Avoid log(0) by filtering out zeros
    Zt = np.sum(pdf_nonzero[:, None]**(1/T), axis = 0) * dx * dy
    
    lnpy = - 241.08 + 356.79 #Evidence for union data set with same priors calculated by Multinest
    
    if not with_out_evidence:
        lnZt = np.log(Zt) + lnpy / T
    else:
        lnZt = np.log(Zt)
    
    #Plot if desired
    if plot == True:
        # Configure Matplotlib to use LaTeX for text rendering
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        plt.rcParams.update({'font.size': 16})
        plt.semilogx()
        plt.plot(T, lnZt)
        plt.xlabel('$T$')
        plt.ylabel('ln $Z[T]$')
        if save == True:
            plt.savefig(path_to_save+'Zt_posterior_estimate.pdf', dpi=300)
        plt.show()
    return lnZt, pdf_values, [dx, dy]

#Temperature dependant partition function via flow as a function of J
def plot_FreeEnergy_TJ(inn, DEVICE, N_DIM: int, nsamples: int = 10000, direction: int = 0, save: bool = False, path_to_save: str = None) -> None:

    #Create 2d grid
    T = np.linspace(0.1, 50, 100)
    J = np.linspace(-39, 39, 100) #Larger J than J = np.linspace(-39, 39, 100) results into numerical overflow in direction 1 (i.e. J_2)

    #Sample from the gaussian
    alphas = torch.randn(nsamples, N_DIM).to(DEVICE).double()
    thetas, det_theta_alpha = inn(alphas, rev=True)
    alphas, thetas, det_theta_alpha= alphas.cpu().detach().numpy(), thetas.cpu().detach().numpy(), det_theta_alpha.cpu().detach().numpy()

    #Precomputations
    alpha_sum = np.sum(alphas**2, axis=1)
    exp_alpha = np.exp(-1/2 * alpha_sum)
    
    exp_alpha_term = (exp_alpha[:,None]**(1/T-1))
    exp_det_term = np.exp(det_theta_alpha)[:,None]**(1-1/T)
    exp_theta_term = np.exp(((1/T[:, None])*J)[None, :,:]*thetas[:,direction, None, None])
    
    log_sum = np.log(np.sum(exp_alpha_term[:,:,None] * exp_det_term[:,:,None] * exp_theta_term, axis = 0))
    
    lnpy = - 241.08 + 356.79
    
    #Free energy
    F = - lnpy + T[:, None] * np.log(nsamples) + (1-T[:, None]) * np.log(2 * np.pi) - T[:, None] * log_sum
    
    F = - F / lnpy
        
    #Print F at T = 1, J = 0
    T1, J0 = np.argmin(np.abs(T - 1)), np.argmin(np.abs(J))
    print("T=1", T[T1], "J=0", J[J0], "F", F[T1, J0])

    #Plotting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams.update({'font.size': 20})
        
    plt.figure(figsize=(10, 10))
    plt.imshow(F.T,cmap='viridis',aspect='auto',origin='lower',extent=[T.min(), T.max(), J.min(), J.max()])
    plt.colorbar(label='Free energy', location = 'top')
    plt.xlabel('$T$')
    plt.ylabel('$J_'+str(direction + 1)+'$')

    levels = np.linspace(F.min(), F.max(), 20)  # Define levels for the contours
    plt.contour(T.flatten(), J.flatten(), F.T, levels=levels, colors='white',linewidths=0.5)
    
    plt.plot(T[T1], J[J0], 'black', marker='o', markersize=10, label = '$T=1$ and '+'$J_'+str(direction + 1)+'=0$')
    plt.legend(loc = 'lower right')
    if save:
        plt.savefig(path_to_save+'FreeEnergy_TJ.pdf', dpi = 300, bbox_inches='tight')
    plt.show()
    return None

#Temperature dependant partition function via flow as a function of J
def presi_plot_FreeEnergy_TJ(inn, DEVICE, N_DIM: int, nsamples: int = 10000, direction: int = 0, save: bool = False, path_to_save: str = None) -> None:

    #Create 2d grid
    T = np.linspace(0.1, 50, 100)
    J = np.linspace(-39, 39, 100) #Larger J than J = np.linspace(-39, 39, 100) results into numerical overflow in direction 1 (i.e. J_2)

    #Sample from the gaussian
    alphas = torch.randn(nsamples, N_DIM).to(DEVICE).double()
    thetas, det_theta_alpha = inn(alphas, rev=True)
    alphas, thetas, det_theta_alpha= alphas.cpu().detach().numpy(), thetas.cpu().detach().numpy(), det_theta_alpha.cpu().detach().numpy()

    #Precomputations
    alpha_sum = np.sum(alphas**2, axis=1)
    exp_alpha = np.exp(-1/2 * alpha_sum)
    
    exp_alpha_term = (exp_alpha[:,None]**(1/T-1))
    exp_det_term = np.exp(det_theta_alpha)[:,None]**(1-1/T)
    exp_theta_term = np.exp(((1/T[:, None])*J)[None, :,:]*thetas[:,direction, None, None])
    
    log_sum = np.log(np.sum(exp_alpha_term[:,:,None] * exp_det_term[:,:,None] * exp_theta_term, axis = 0))
    
    lnpy = - 241.08 + 356.79
    
    #Free energy
    F = - lnpy + T[:, None] * np.log(nsamples) + (1-T[:, None]) * np.log(2 * np.pi) - T[:, None] * log_sum
    
    F = - F / lnpy
        
    #Print F at T = 1, J = 0
    T1, J0 = np.argmin(np.abs(T - 1)), np.argmin(np.abs(J))
    print("T=1", T[T1], "J=0", J[J0], "F", F[T1, J0])

    #Plotting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams.update({'font.size': 27})
        
    plt.figure(figsize=(16, 10))
    plt.imshow(F.T,cmap='viridis',aspect='auto',origin='lower',extent=[T.min(), T.max(), J.min(), J.max()])
    plt.colorbar(label='Free energy', location = 'left')
    plt.xlabel('$T$')
    plt.ylabel('$J_'+str(direction + 1)+'$')

    levels = np.linspace(F.min(), F.max(), 20)  # Define levels for the contours
    plt.contour(T.flatten(), J.flatten(), F.T, levels=levels, colors='white',linewidths=0.5)
    
    plt.plot(T[T1], J[J0], 'black', marker='o', markersize=10, label = '$T=1$ and '+'$J_'+str(direction + 1)+'=0$')
    plt.legend(loc = 'lower right')
    if save:
        plt.savefig(path_to_save+'FreeEnergy_TJ_presi.pdf', dpi = 300, bbox_inches='tight')
    plt.show()
    return None

#Free energy as a function of J
def plot_FreeEnergy_JJ(inn, DEVICE, N_DIM: int, nsamples: int = 10000, T: int = 1, save: bool = False, path_to_save: str = None) -> None:

    #Create 2d grid
    J_1 = np.linspace(-50, 50, 200) #The larger J, the more difficult to find Js = 0
    J_2 = np.linspace(-50, 50, 200) #The larger J, the more difficult to find Js = 0

    #Sample from the gaussian
    alphas = torch.randn(nsamples, N_DIM).to(DEVICE).double()
    thetas, det_theta_alpha = inn(alphas, rev=True)
    alphas, thetas, det_theta_alpha= alphas.cpu().detach().numpy(), thetas.cpu().detach().numpy(), det_theta_alpha.cpu().detach().numpy()

    #Precomputations
    alpha_sum = np.sum(alphas**2, axis=1)
    exp_alpha = np.exp(-1/2 * alpha_sum)

    exp_alpha_term = (exp_alpha**(1/T-1)) #[nsamples]
    exp_det_term = np.exp(det_theta_alpha)**(1-1/T) #[nsamples]
    exp_theta_term = np.exp(-(1/T)* ((J_1 * thetas[:,0, None])[:, :, None] + (J_2 * thetas[:,1, None])[:, None, :])) #[nsamples, J_1, J_2]
     
    log_sum = np.log(np.sum(exp_alpha_term[:,None,None] * exp_det_term[:,None,None] * exp_theta_term, axis = 0)) #[J_1, J_2]

    lnpy = - 241.08 + 356.79
    
    #Free energy
    F = - lnpy + T * np.log(nsamples) + (1-T) * np.log(2 * np.pi) - T * log_sum
    
    F = - F / lnpy
        
    #Print F at T = 1, J = 0
    J0_1, J0_2 = np.argmin(np.abs(J_1)), np.argmin(np.abs(J_2))
    print("J_1 = 0", J_1[J0_1], "J_2 = 0", J_2[J0_2], "F", F[J0_1, J0_2])

    #Plotting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams.update({'font.size': 20})
    
    plt.figure(figsize=(10, 10))
    plt.imshow(F.T,cmap='viridis',aspect='auto',origin='lower',extent=[J_1.min(), J_1.max(), J_2.min(), J_2.max()])
    plt.colorbar(label='Free Energy', location = 'top')
    plt.xlabel('$J_1$')
    plt.ylabel('$J_2$')
    
    plt.plot(1, 0, 'black', marker='o', markersize=10, label = '$J_1=0$ and $J_2=0$')
    plt.legend(loc = 'lower right')
    levels = np.linspace(F.min(), F.max(), 20)  # Define levels for the contours
    plt.contour(J_1.flatten(), J_2.flatten(), F.T, levels=levels, colors='white',linewidths=0.5)
    if save:
        plt.savefig(path_to_save+'FreeEnergy_JJ.pdf', dpi = 300, bbox_inches='tight')
    plt.show()
    return None

#Calculate free energy at certain T and J
def free_energy(inn, DEVICE, N_DIM: int, T, J_1, J_2, nsamples = 10000):
    #Sample from the gaussian
    alphas = torch.randn(nsamples, N_DIM).to(DEVICE).double()
    thetas, det_theta_alpha = inn(alphas, rev=True)
    alphas, thetas, det_theta_alpha= alphas.cpu().detach().numpy(), thetas.cpu().detach().numpy(), det_theta_alpha.cpu().detach().numpy()

    #Precomputations
    alpha_sum = np.sum(alphas**2, axis=1)
    exp_alpha = np.exp(-1/2 * alpha_sum)
    
    exp_alpha_term = exp_alpha**(1/T-1)
    exp_det_term = np.exp(det_theta_alpha)**(1-1/T)
    exp_theta_term = np.exp(-(1/T)* ((J_1 * thetas[:,0]) + (J_2 * thetas[:,1])))
    
    log_sum = np.log(np.sum(exp_alpha_term * exp_det_term * exp_theta_term, axis = 0))
    
    lnpy = - 241.08 + 356.79
    
    #Free energy
    F = - lnpy + T * np.log(nsamples) + (1-T) * np.log(2 * np.pi) - T * log_sum
    return F

#Calculate free energy time derivative, i.e. entropy
def der_FT(inn, DEVICE, N_DIM: int, nsamples: int = 10000, T_initial: float = 1., plot: bool = True, save: bool = False, path_to_save: str = None) -> float:
    T = np.linspace(T_initial-0.05, T_initial+0.05, 100)
    
    alphas = torch.randn(nsamples, N_DIM).to(DEVICE).double()
    thetas, det_theta_alpha = inn(alphas, rev=True)
    alphas, thetas, det_theta_alpha= alphas.cpu().detach().numpy(), thetas.cpu().detach().numpy(), det_theta_alpha.cpu().detach().numpy()
    
    #Precomputations
    alpha_sum = np.sum(alphas**2, axis=1)
    exp_alpha = np.exp(-1/2 * alpha_sum)
    
    exp_alpha_term = (exp_alpha[:,None]**(1/T-1).T)
    exp_det_term = np.exp(det_theta_alpha)[:,None]**(1-1/T).T
    
    log_sum = np.log((exp_alpha_term * exp_det_term).sum(axis = 0))

    #Evidence from Multinest
    lnpy = - 241.08 + 356.79
    
    F = - lnpy + T * np.log(nsamples) + (1-T) * np.log(2 * np.pi) - T * log_sum
    minus_F = - F
        
    coefficients = np.polyfit(T, minus_F, 1)  # 1 means linear
    entropy, intercept = coefficients
    print("Entropy:", entropy)
    
    #Plotting
    if plot:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        plt.rcParams.update({'font.size': 16})
    plt.plot(T, entropy*T + intercept, label = 'linear fit')
    plt.plot(T,minus_F, label = 'Free energy')
    plt.xlabel('T')
    plt.ylabel('- F[T]')
    plt.plot(1, lnpy, 'ro', label = 'Bayesian evidence')
    plt.legend(loc = 'upper right')
    if save:
        plt.savefig(path_to_save+'der_FT_inn.pdf', dpi = 300)
    plt.show()
    return entropy