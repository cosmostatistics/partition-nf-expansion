import torch
import numpy as np
from scipy.stats import gaussian_kde
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

    #Push Gaussian samples through INN
    alphas = torch.randn(nsamples, N_DIM).to(DEVICE).double()
    thetas, logdetthetaalpha = inn(alphas, rev=True)
    alphas, thetas, logdetthetaalpha = alphas.cpu().detach().numpy(), thetas.cpu().detach().numpy(), logdetthetaalpha.cpu().detach().numpy()
    
    return - 1/nsamples * np.sum(log_standard_normal2d(alphas)) + 1/nsamples * np.sum(logdetthetaalpha)

#Analtyical entropy for Gaussian toy model
def entropy_ana_gaussian(cov: np.ndarray, N_DIM :int) -> float:
    return np.log(2*np.pi) + 0.5*np.log(np.linalg.det(cov)) + 0.5*N_DIM