import numpy as np
import matplotlib.pyplot as plt
import torch

from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

#Choose device
DEVICE = torch.device('cpu')

#Function to obtain the jacobian and its determinant at a specific point
def jacobian_cal(n_flow, nsamples: int = 10000,ndim: int = 2,sampling: bool = True,point: np.ndarray = [0.,0.]) -> np.ndarray:

    if sampling == True:
        alphas = torch.randn(nsamples, ndim, requires_grad=True).to(DEVICE).to(torch.float64)
        #print(alphas.shape)
        thetas, logdet = n_flow(alphas, rev=True)
    elif sampling == False:
        alphas = torch.tensor([point], requires_grad=True).to(DEVICE).to(torch.float64)
        thetas, logdet = n_flow(alphas, rev=True)
        nsamples = 1

    jacobian_1 = torch.autograd.grad(thetas[0][0], alphas, create_graph=True)
    jac1 = np.reshape(jacobian_1[0].cpu().detach().numpy(), (nsamples,ndim))
    jacobian_2 = torch.autograd.grad(thetas[0][1], alphas, create_graph=True)
    jac2 = np.reshape(jacobian_2[0].cpu().detach().numpy(), (nsamples,ndim))
    jacobian = jac1[:,:,None] + jac2[:,None,:]

    return jacobian,logdet

#Function to decompose the jacobian into the basis matrices
def decomposition_jacobian(jacobian: np.ndarray, i:int = 0) -> np.ndarray:
    # Basis matrices
    M1 = np.array([[1, 0], [0, 1]])
    M2 = np.array([[1, 1], [0, -1]])
    M3 = np.array([[0, 1], [1, 0]])
    M4 = np.array([[0, 1], [-1, 0]])

    basis_matrices = np.array([M1.flatten(), M2.flatten(), M3.flatten(), M4.flatten()])
  
    A = jacobian
    A_flat = A.flatten()
    # Compute the coefficients of the basis matrices
    coefficients = np.linalg.lstsq(basis_matrices.T, A_flat, rcond=None)[0]

    kappa, gamma_1, gamma_2, omega = coefficients
    print(f"Kappa: {kappa}, Gamma_1: {gamma_1}, Gamma_2: {gamma_2}, Omega: {omega}")

#Visulaising the geometrical volume change
def geometrical_volume_change(n_flow, path_to_save_plots: str, nsamples: int = 10000, gridsize: int = 20, N_DIM: int = 2) -> None:

    #Create 2d grid on alpha space
    alpha1_lin = np.linspace(-2, 2, gridsize)
    alpha2_lin = np.linspace(-2, 2, gridsize)
    alpha1, alpha2 = np.meshgrid(alpha1_lin, alpha2_lin)
    grid = np.vstack([alpha1.ravel(), alpha2.ravel()]).T

    #Transform to theta space
    grid = torch.tensor(grid).to(DEVICE).double()
    theta_grid, logdet_grid = n_flow(grid, rev=True)
    theta_grid = theta_grid.cpu().detach().numpy()
    logdet_grid = logdet_grid.cpu().detach().numpy()
    theta1 = theta_grid[:,0].reshape(gridsize,gridsize)
    theta2 = theta_grid[:,1].reshape(gridsize,gridsize)

    #Reverse inn
    alphas = torch.randn(nsamples, N_DIM).to(DEVICE).double()
    thetas, _ = n_flow(alphas, rev=True)
    alphas, thetas= alphas.cpu().detach().numpy(), thetas.cpu().detach().numpy()
    
    # Plotting settings
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams.update({'font.size': 24})

    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

    # === Left subplot: Transformed posterior ===
    h1 = ax1.hist2d(thetas[:, 0], thetas[:, 1], bins=100, cmap='viridis')
    for i in range(len(alpha1_lin)):
        ax1.plot(theta1[i, :], theta2[i, :], color='white', lw=0.5)
        ax1.plot(theta1[:, i], theta2[:, i], color='white', lw=0.5)
    ax1.set_xlim(theta1.min(), theta1.max())
    ax1.set_ylim(theta2.min(), theta2.max())
    ax1.set_xlabel(r'$\theta_1$')
    # Place the y-axis label and ticks on the right side
    ax1.yaxis.set_label_position('right')
    ax1.yaxis.tick_right()
    ax1.set_ylabel(r'$\theta_2$')

    # Plotting a specific box on the theta grid
    griddy = gridsize // 3
    bottom_left = [theta1[griddy, griddy], theta2[griddy, griddy]]
    bottom_right = [theta1[griddy+1, griddy], theta2[griddy+1, griddy]]
    top_right = [theta1[griddy+1, griddy+1], theta2[griddy+1, griddy+1]]
    top_left = [theta1[griddy, griddy+1], theta2[griddy, griddy+1]]
    poly = np.vstack([bottom_left, bottom_right, top_right, top_left, bottom_left])
    polygon = Polygon(poly, closed=True, linewidth=2.5, edgecolor='red', fill=False, facecolor='red')
    ax1.add_patch(polygon)

    # === Right subplot: Original grid ===
    h2 = ax2.hist2d(alphas[:, 0], alphas[:, 1], bins=100, cmap='viridis')
    for i in range(len(alpha1_lin)):
        ax2.plot(alpha1[i, :], alpha2[i, :], color='white', lw=0.5)
        ax2.plot(alpha1[:, i], alpha2[:, i], color='white', lw=0.5)
    ax2.set_xlim(alpha1_lin[0], alpha1_lin[-1])
    ax2.set_ylim(alpha2_lin[0], alpha2_lin[-1])
    ax2.set_xlabel(r'$\alpha_1$')
    ax2.set_ylabel(r'$\alpha_2$')

    # Plotting a specific box on the alpha grid
    a0, a1 = alpha1[griddy, griddy], alpha2[griddy, griddy]
    width = alpha1[griddy+1, griddy+1] - alpha1[griddy, griddy]
    height = alpha2[griddy+1, griddy+1] - alpha2[griddy, griddy]
    rect = plt.Rectangle((a0, a1), width, height, linewidth=3, edgecolor='red', fill=False, facecolor='red')
    ax2.add_patch(rect)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the combined figure if needed
    plt.savefig(path_to_save_plots + 'combined_grid_transformation.png', dpi=300, bbox_inches='tight')

    plt.show()

    #Logdet at specific box and decomposition of jacobian
    griddy = gridsize//3
    bottom_left_det = [logdet_grid[griddy**2]]
    print(f"Logdet at specific box: {bottom_left_det[0]}")
    area = ConvexHull(poly).volume
    area_rect = width*height
    print(f"Area of specific box in theta space: {area}")
    print(f"Area of specific box in alpha space: {area_rect}")

    point = [a0+width/2,a1+height/2]
    jacobian,_ = jacobian_cal(n_flow, nsamples, sampling=False,point=point)
    jacobian = jacobian[0]
    print(f"Jacobian at specific box: {jacobian}")
    decomposition_jacobian(jacobian)