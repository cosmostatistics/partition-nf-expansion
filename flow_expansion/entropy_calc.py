import torch
import numpy as np
from scipy.stats import gaussian_kde
import scipy.special as scsp
from typing import List, Tuple
import matplotlib.pyplot as plt

#Estimate entroy via histogram
def entropy_hist_estimate(samples: np.ndarray, bins: int = 60) -> float:
    # Compute the 2D histogram with counts
    hist_post, x_edges, y_edges = np.histogram2d(*samples.T, bins=bins, density=False)

    # Calculate the bin widths in x and y directions, thus bin area
    x_width = np.diff(x_edges).mean()
    y_width = np.diff(y_edges).mean()
    bin_area = x_width * y_width

    #Use only non-zero values and normalisse
    nonzero_hist = hist_post[hist_post > 0]
    nonzero_hist = nonzero_hist / np.sum(nonzero_hist) 
    return -np.sum(nonzero_hist * np.log(nonzero_hist)) + np.log(bin_area)

#Estimate entropy via Kernel Density Estimate
def entropy_kde_estimate(samples: np.ndarray, grid_size: int = 100, plot: bool = True) -> float:
    # Perform 2D KDE on the samples
    kde = gaussian_kde(samples.T)  # Note: samples should be transposed for gaussian_kde
    
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
    
    # Calculate the entropy using the estimated PDF
    pdf_nonzero = pdf_values[pdf_values > 0]  # Avoid log(0) by filtering out zeros
    entropy = -np.sum(pdf_nonzero * np.log(pdf_nonzero)) * dx * dy
    
    #Plot if desired
    if plot == True:
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(X, Y, pdf_values, shading='auto', cmap='viridis')
        plt.colorbar(label='Estimated PDF')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'2D KDE with entropy: {entropy:.3f}')
        plt.show()
    
    return entropy

#Estime entropy via normalising flow
def entropy_flow(inn, DEVICE, N_DIM: int, nsamples: int = 125000) -> float:
    #Define logarithm of standard normal distribution
    def log_standard_normal2d(alpha: np.ndarray) -> np.ndarray:
        return -0.5*np.sum(alpha**2, axis=1) - np.log(2*np.pi)

    #Run Gaussian samples through INN
    alphas = torch.randn(nsamples, N_DIM).to(DEVICE).double()
    thetas, logdetthetaalpha = inn(alphas, rev=True)
    alphas, thetas, logdetthetaalpha = alphas.cpu().detach().numpy(), thetas.cpu().detach().numpy(), logdetthetaalpha.cpu().detach().numpy()
    
    return - 1/nsamples * np.sum(log_standard_normal2d(alphas)) + 1/nsamples * np.sum(logdetthetaalpha)

#Analtyical entropy for Gaussian
def entropy_ana_gaussian(cov: np.ndarray, N_DIM :int) -> float:
    return np.log(2*np.pi) + 0.5*np.log(np.linalg.det(cov)) + 0.5*N_DIM

# ### WARUM SIND DIE HIER DRIN?

# #Calculate moments from INN
# def moment_inn(inn, cumulant: List, N_DIM: int, DEVICE, nsamples: int = 125000) -> float:
#     alphas = torch.randn(nsamples, N_DIM).to(DEVICE).double()
#     thetas, logdetthetaalpha = inn(alphas, rev=True)
#     alphas, thetas, logdetthetaalpha = alphas.cpu().detach().numpy(), thetas.cpu().detach().numpy(), logdetthetaalpha.cpu().detach().numpy()
#     return 1 / nsamples * np.sum(thetas[:, 0]**cumulant[0] * thetas[:, 1]**cumulant[1])

# def mean_var_skew_kurt_from_moments(moments: List, print_out: bool = True) -> np.ndarray:
#     #Calculate nth central moment
#     def nth_central_moment(n: int, moments: List) -> float:
#         assert len(moments) > n-1, "Not enough moments calculated"
#         sum = 0
#         for k in range(0, n+1):
#             sum += scsp.binom(n, k) * moments[k]* moments[1]**(n-k) * (-1)**(n-k)
#         return sum

#     #Calculate central moments
#     def central_moments_calc(order_moment: int, moments: List) -> List:
#         nth_central_moments = [nth_central_moment(i, moments) for i in range(order_moment)]
#         return nth_central_moments

#     #Calculate nth standardized moment
#     def nth_standardized_moment(n: int, central_moments: List) -> float:
#         assert len(central_moments) > n-1, "Not enough cumulants calculated"
#         if n == 2:
#             return 1
#         else:
#             return central_moments[n] / central_moments[2]**(n/2)
        
#     #Calculate standardized moments
#     def standardized_moments_calc(order_moment: int, nth_central_moments = None) -> List:
#         nth_standardized_moments = [nth_standardized_moment(i, nth_central_moments) for i in range(order_moment)]
#         return nth_standardized_moments

#     moments_long = np.ones(len(moments)+1)
#     moments_long[1:] = moments
#     moments = moments_long
#     order_moment = len(moments)
#     nth_central_moments = central_moments_calc(order_moment, moments)
#     nth_standardized_moments = standardized_moments_calc(order_moment, nth_central_moments)
#     mean = moments[1]
#     var = nth_central_moments[2]
#     skew = nth_standardized_moments[3]
#     kurt = nth_standardized_moments[4]
#     if print_out:
#         print("Mean Value:",moments[1],"\n","Variance:", nth_central_moments[2],"\n", "Skewness:", nth_standardized_moments[3],"\n", "Kurtosis:", nth_standardized_moments[4])
#     #return [mean, var, skew, kurt]
#     return np.array([mean, var, skew, kurt])

# def mean_var_skew_kurt_theta1_theta2(inn_moments_theta1: np.ndarray, inn_moments_theta2: np.ndarray) -> List[np.ndarray]:
#     n_test = inn_moments_theta1.shape[1]
#     quantities_theta1 = np.empty((4, n_test))
#     quantities_theta2 = np.empty((4, n_test))
#     for j in range(n_test):
#         quantities_theta1[:, j] = mean_var_skew_kurt_from_moments(inn_moments_theta1[:, j], print_out=False)
#         quantities_theta2[:, j] = mean_var_skew_kurt_from_moments(inn_moments_theta2[:, j], print_out=False)
#     return quantities_theta1, quantities_theta2