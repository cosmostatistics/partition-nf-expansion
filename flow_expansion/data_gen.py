import os
import matplotlib.pyplot as plt
from typing import List
import numpy as np

#Accumulation function for the data generation
def generate_data(model: str, params: List = None, dim: int = 2, nsamples: int = 10000, plot: bool = False, save: bool = False, path_to_files: str = ""):
    if model == "gaussian":
        samples = generate_gaussian(params, dim, nsamples, plot, save, path_to_files)
    elif model == "offdiag_gaussian":
        offdiagonal = True
        samples = generate_gaussian(params, dim, nsamples, plot, save, path_to_files, offdiagonal)
    elif model == "multimodal_gaussian":
        samples = generate_mutimodal_gaussian(params, dim, nsamples, plot, save, path_to_files)
    else:
        raise ValueError("Model not implemented")
    return samples

#Generate Gaussian samples
def generate_gaussian(params: List, dim: int, nsamples: int, plot: bool = False, save: bool = False, path_to_files: str = "", offdiagonal: bool = False):
    assert len(params) == 2, "The parameters for the Gaussian model are mu and sigma"
    if dim == 1:
        mean, sigma = params[0], params[1]
        samples = np.random.normal(mean, sigma, nsamples).reshape((nsamples, 1))
    elif dim > 1:
        if not offdiagonal:
            mean, sigma = params[0], params[1]
            mean_normal = np.ones(dim) * mean
            cov_normal = np.eye(dim) * sigma
        else:
            try:
                mean_normal = params[0]
                cov_normal = params[1]
            except:
                raise ValueError("Parameters are not provided in the correct format")
        samples = np.random.multivariate_normal(mean = mean_normal, cov = cov_normal, size = nsamples)
    else:
        raise ValueError("The dimension of the Gaussian model must be greater than 0")
    if plot:
        # Configure Matplotlib to use LaTeX for text rendering
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        plt.rcParams.update({'font.size': 14})
    
        if not offdiagonal:
            samples_plot = samples[:, 0]
            hist = plt.hist(samples_plot, bins=100, density=True)
            x_lim = hist[1][[0, -1]]
            plt.xlim(x_lim)
            plt.xlabel("x")
            plt.ylabel("Density")
            plt.title("Plot of Gaussian samples")
            if save:
                os.makedirs(path_to_files, exist_ok=True)
                plt.savefig(path_to_files+"gaussian_samples.pdf", dpi=300)
            plt.show()
        else:
            plt.figure(figsize=(8, 6))
            plt.hist2d(*samples.T, bins=50, cmap='plasma')
            plt.colorbar(label='Counts')
            plt.title('2d offdiagonal Gaussian')
            plt.xlabel('x1')
            plt.ylabel('x2')
            if save:
                os.makedirs(path_to_files, exist_ok=True)
                plt.savefig(path_to_files+"gaussian_samples.pdf", dpi=300)
            plt.show()
    if save:
        os.makedirs(path_to_files, exist_ok=True)
        np.save(path_to_files+"samples_gaussian.npy", samples)
        if not offdiagonal:
            np.save(path_to_files+"params_gaussian.npy", params)
        else:
            np.save(path_to_files+"mean_gaussian.npy", params[0])
            np.save(path_to_files+"cov_gaussian.npy", params[1])
    return samples

#Generate multimodal Gaussian samples
def generate_mutimodal_gaussian(params: List, dim: int, nsamples: int, plot: bool = False, save: bool = False, path_to_files: str = ""):
    #Params have to be a list of list with mean and cov each: [[np.array([0, 0]), np.array([[2, 1],[1, 1]])], [np.array([4, 4]), np.array([[1, 0],[0, 1]])]]
    try:
        params1 = params[0]
        params2 = params[1]
        samples1 = generate_gaussian(params1, dim, int(nsamples/2), False, False, path_to_files, True)
        samples2 = generate_gaussian(params2, dim, int(nsamples/2), False, False, path_to_files, True)
    except:
        raise ValueError("Parameters are not provided in the correct format")
    samples = np.concatenate((samples1, samples2), axis=0)
    if plot:
        plt.figure(figsize=(8, 6))
        plt.hist2d(*samples.T, bins=50, cmap='plasma')
        plt.colorbar(label='Counts')
        plt.title('2d multimodal Gaussian')
        plt.xlabel('x1')
        plt.ylabel('x2')
        if save:
            os.makedirs(path_to_files, exist_ok=True)
            plt.savefig(path_to_files+"multimodal_gaussian_samples.pdf", dpi=300)
        plt.show()
    if save:
        os.makedirs(path_to_files, exist_ok=True)
        np.save(path_to_files+"samples_multimodal_gaussian.npy", samples)
        np.save(path_to_files+"mean_gaussian1.npy", params1[0])
        np.save(path_to_files+"cov_gaussian1.npy", params1[1])
        np.save(path_to_files+"mean_gaussian2.npy", params2[0])
        np.save(path_to_files+"cov_gaussian2.npy", params2[1])
    return samples