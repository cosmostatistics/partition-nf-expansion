# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

#Torch imports
import torch 
import torch.nn as nn

#Standard imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
from typing import List

#Function yields Vanilla INN (similarly to FreIA documentation)
def inn_vanilla(N_DIM: int, N_LAYERS: int, WIDTH_SUBNET: int, permute_soft: bool):
    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, WIDTH_SUBNET), nn.GELU(),
                         nn.Linear(WIDTH_SUBNET,  dims_out))
    inn = Ff.SequenceINN(N_DIM)
    for k in range(N_LAYERS):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=permute_soft)
    return inn


def load_trained_inn(path_to_files: str):
    #Parameters in Capital Letters need to be initialised as global variables
    try:
        with open(path_to_files + 'param_dict.pkl', 'rb') as f:
            param_dict = pickle.load(f)
        for key, value in param_dict.items():
            globals()[key] = value
        if NETWORK == 'vanilla_inn':
            if N_DIM == 1:
                permute_soft = False
                print("Permute_soft set to False for 1d")
            else:
                permute_soft = PERMUTE_SOFT
            inn = inn_vanilla(N_DIM, N_LAYERS, WIDTH_SUBNET, permute_soft)
            inn.to(torch.double)
        else:
            raise ValueError("Network not implemented")
        inn.load_state_dict(torch.load(path_to_files+'inn_state_dict.pth'))
        print("Inn loaded, trained with following parameters: ", param_dict)
        return inn
    except FileNotFoundError:
        print("File not found")
        return None

#Function trains INN
def train_inn(DATA: np.ndarray, NETWORK: float, BATCHSIZE: int, N_DIM: int, N_LAYERS: int, WIDTH_SUBNET: int, LEARNING_RATE: float, NUM_EPOCS: int, DEVICE: str, PERMUTE_SOFT: bool = True, plot: bool = True, save: bool = False, path_to_files: str = ""):
    if NETWORK == 'vanilla_inn':
        if N_DIM == 1:
            permute_soft = False
            print("Permute_soft set to False for 1d")
        else:
            permute_soft = PERMUTE_SOFT
        inn = inn_vanilla(N_DIM, N_LAYERS, WIDTH_SUBNET, permute_soft)
    else:
        raise ValueError("Network not implemented")
    inn.to(DEVICE)
    
    #Define parameter dictionary for saving
    param_dict = {"NETWORK": NETWORK, "BATCHSIZE": BATCHSIZE, "N_DIM": N_DIM, "N_LAYERS": N_LAYERS, "LEARNING_RATE": LEARNING_RATE, "WIDTH_SUBNET": WIDTH_SUBNET, "NUM_EPOCS": NUM_EPOCS, "DEVICE": DEVICE, "PERMUTE_SOFT": PERMUTE_SOFT}

    #Choose Adam optimizer
    optimizer = torch.optim.Adam(inn.parameters(), lr=LEARNING_RATE)
    
    #Create DataLoaders
    samples = torch.tensor(DATA, dtype=torch.double)
    dataset = torch.utils.data.TensorDataset(samples)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=True)

    #Define loss arrays
    training_loss = []
    validation_loss = []
    
    #Training loop
    for epoch in tqdm(range(NUM_EPOCS)):
        #Training part
        inn.to(torch.double)
        inn.train()
        training_meanwhile = []
        for i, (data,) in enumerate(train_loader):
            optimizer.zero_grad()
            x = data.to(DEVICE)
            z, log_jac_det = inn(x)
            loss = loss_criterion(z, log_jac_det, N_DIM)
            loss.backward()
            optimizer.step()
            training_meanwhile.append(loss.detach())
        training_loss.append(np.mean(training_meanwhile))
    
        #Validation part
        with torch.no_grad():
            validation_meanwhile = []
            for i, (data,) in enumerate(val_loader):
                x = data.to(DEVICE)
                z, log_jac_det = inn(x)
                loss = loss_criterion(z, log_jac_det, N_DIM)
                validation_meanwhile.append(loss.detach())
            validation_loss.append(np.mean(validation_meanwhile))
    
    #Save model and parameters
    if save:
        with open(path_to_files + 'param_dict.pkl', 'wb') as f:
            pickle.dump(param_dict, f)
        if DEVICE == torch.device('cuda'):
            inn.to(torch.device('cpu'))
        torch.save(inn.state_dict(), path_to_files+'inn_state_dict.pth')
    
    if plot:    
        # Configure Matplotlib to use LaTeX for text rendering
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        plt.rcParams.update({'font.size': 14})
    
        plt.title('Training Loss')
        plt.plot(training_loss)
        plt.xlabel('Training Steps')
        plt.ylabel('Negative Log Likelihood')
        if save:
            plt.savefig(path_to_files+"training_loss.pdf", dpi=300)
        plt.show()
    
        plt.title('Validation Loss')
        plt.plot(validation_loss)#.mean(axis = -1))
        plt.xlabel('Validation Steps')
        plt.ylabel('Negative Log Likelihood')
        if save:
            plt.savefig(path_to_files+"validation_loss.pdf", dpi=300)
        plt.show()
    return inn

#Define loss function
def loss_criterion(z, log_jac_det, N_DIM):
    loss = 0.5*torch.sum(z**2, 1) - log_jac_det
    loss = torch.mean(loss) / N_DIM
    return loss
    
#Plot trained INN
def plot_results(inn, nsamples: int, N_DIM: int, samples: np.ndarray, save: bool = True, path_to_files: str = "", direction: int = 0, NETWORK: str = 'vanilla_inn', visualize2d: bool = False):
    with torch.no_grad():
        z = torch.randn(nsamples, N_DIM, dtype=torch.double)
        net_out, _ = inn(z, rev=True)
        samples_pred = net_out.detach().numpy()
        if not visualize2d:
            samples_pred_plot = samples_pred[:, direction]
        else:
            assert N_DIM == 2, "Only implemented for 2D"
            samples_pred_plot = samples_pred
    
    # Configure Matplotlib to use LaTeX for text rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams.update({'font.size': 14})
    
    # plot samples
    bins = 100
    if N_DIM > 1 and not visualize2d:
       samples_plot = samples[:, direction]
    else:
        samples_plot = samples
    if not visualize2d:
        hist = plt.hist(samples_plot, bins=100, density=True, alpha=0.5, label='samples')
        x_lim = hist[1][[0, -1]]
        x_lim = [x_lim[0] - 0.1*abs(x_lim[0]), x_lim[1] + 0.1*abs(x_lim[1])]
        plt.xlim(x_lim)
        plt.title("Plot of samples vs INN samples")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.hist(samples_pred_plot, density=True, bins = bins, alpha=0.5, label="$INN(z)$")
        plt.legend()
        if save:
            plt.savefig(path_to_files+"samples_vs_inn_samples.pdf", dpi=300)
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        hist1 = ax1.hist2d(*samples_pred_plot.T, bins=100, cmap='plasma')
        ax1.set_title('INN samples')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        hist2 = ax2.hist2d(*samples_plot.T, bins=100, cmap='plasma')
        ax2.set_title('Samples')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        fig.suptitle('Comparison of "true" samples and INN samples')
        # cbar = fig.colorbar(hist1[3], ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.04)
        # cbar.set_label('Counts')
        plt.tight_layout()#(rect=[0, 0, 1, 0.95])
        if save:
            plt.savefig(path_to_files+"samples_vs_inn_samples.pdf", dpi=300)
        plt.show()
    return samples_pred
    
#Calculate derivatives
def derivatives(inn, max_der: int, ndim: int) -> np.ndarray:
    # At position zero is value of function evaluated at zero
    alpha0 = torch.zeros(ndim, requires_grad=True, dtype=torch.double).reshape(1, ndim) 
    net_out, _ = inn(alpha0, rev=True)
    gradients = [net_out[0][0]]
    gradients_np = np.empty(max_der)
    gradients_np[0] = net_out[0][0].detach().numpy()
        
    for i in tqdm(range(max_der-1)):
        grad = torch.autograd.grad(gradients[-1], alpha0, create_graph=True, retain_graph= True)[0][0][0]
        gradients.append(grad)
        gradients_np[i+1] = grad.detach().numpy()
    
    return gradients_np

#Calculate laplacian
def laplacian(inn, func: List[int], order_series: int, order_moment: int, ndim: int, NETWORK = 'inn_vanilla') -> np.ndarray:
    assert ndim == 2, "Laplacian only implemented for 2D"
    
    # At position zero is value of function evaluated at zero
    alpha0 = torch.zeros(ndim, requires_grad=True, dtype=torch.double).reshape(1, 2) 
    if NETWORK == 'normflows_mlp':
        net_out = inn.forward(alpha0)
    else:
        net_out, _ = inn(alpha0, rev=True)
    f_1 = net_out[0][0]
    f_2 = net_out[0][1]
    
    func_val = f_1**func[0] * f_2**func[1]
    
    laplace = [func_val**order_moment]
    laplace_np = np.empty(order_series+1)
    laplace_np[0] = laplace[0].detach().numpy()
    
    for i in range(order_series):

        grad_f = torch.autograd.grad(outputs=laplace[-1], inputs = alpha0, retain_graph=True, create_graph=True)
        intermediate_1 = torch.autograd.grad(outputs=grad_f[0][0][0], inputs = alpha0, retain_graph=True, create_graph=True)[0][0][0]
        intermediate_2 = torch.autograd.grad(outputs=grad_f[0][0][1], inputs = alpha0, retain_graph=True, create_graph=True)[0][0][1]

        laplace.append(intermediate_1+intermediate_2)
        laplace_np[i+1] = laplace[-1].detach().numpy()
    
    return laplace_np