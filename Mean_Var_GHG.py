import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal


class PolicyNetwork(nn.Module):
    def __init__(self, input_size_net, hidden_sizes, output_size_net):
        super(PolicyNetwork, self).__init__()

        # Create a list to hold all layers
        layers = [nn.Flatten(),
                  nn.Linear(input_size_net, hidden_sizes[0]),
                  nn.BatchNorm1d(hidden_sizes[0]),
                  nn.ReLU()]

        for ii in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[ii - 1], hidden_sizes[ii]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size_net))
        layers.append(nn.Softmax(dim=-1))  # Apply softmax along the last dimension

        # Combine all layers into a sequential block
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input x through the entire network
        return self.network(x)


def objective_function(portwealth, portghg_level, rho_coef=0.5, coef_ghg=1):
    # Mean Var
    U = (portwealth - rho_coef * (portwealth - portwealth.mean()) ** 2 - coef_ghg * portghg_level).mean()
    return U


class EralyStopping:
    def __init__(self, patience_es, min_delta=0.0, path='policy_network_deep.pth'):
        self.patience = patience_es
        self.min_delta = min_delta
        self.path = path
        self.wait = 0
        self.best_objective_value = -np.inf
        self.earlystop = False

    def __call__(self, obj_val, model):
        if obj_val > self.best_objective_value:
            self.best_objective_value = obj_val
            self.wait = 0
            self.save_checkpoint(model)
        elif obj_val <= self.best_objective_value + self.min_delta:
            self.wait += 1
            if self.wait > self.patience:
                self.earlystop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

    def load_checkpoint(self, model):
        torch.load(model.state_dict(), self.path)


class OptimizationModel:
    def __init__(self, policy_network, optimizer, n_epochs, minibatchsize, niter, input_size, rho, gamma, network_path,
                 patience, data_in_sample, data_out_of_sample, wealth_init, NRiskyAssets, n_steps, device_p):
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.nepochs = n_epochs
        self.minibatchsize = minibatchsize
        self.niter = niter
        self.input_size = input_size
        self.rho = rho
        self.gamma = gamma
        self.network_path = network_path
        self.patience = patience
        self.simullogreturns_all_IS = data_in_sample['simullogreturns_all']
        self.simulcovariance_IS = data_in_sample['simulcovariance']
        self.ghg_paths_all_IS = data_in_sample['ghg_paths_all']
        self.simullogreturns_all_OOS = data_out_of_sample['simullogreturns_all']
        self.simulcovariance_OOS = data_out_of_sample['simulcovariance']
        self.ghg_paths_all_OOS = data_out_of_sample['ghg_paths_all']
        self.NRiskyAssets = NRiskyAssets
        self.nsteps = n_steps
        self.w0 = wealth_init
        self.device = device_p

    def upper_triangle(self, mat_in):
        ind12 = np.triu(np.array(np.arange(1, self.NRiskyAssets * self.NRiskyAssets + 1)).reshape(self.NRiskyAssets,
                                                                                                  self.NRiskyAssets))
        ind12 = ind12[ind12 != 0] - 1
        mat_out = mat_in.reshape((mat_in.shape[0], mat_in.shape[1] * mat_in.shape[2]))[:, ind12]
        return mat_out

    def prepare_inputs(self, cpath_train, covpaths_train_lag, ghg_path_train_lag, portret_train_in, ghg_level_train_in,
                       tt_train):
        # Apply transformations
        log_ghg_path = np.log1p(ghg_path_train_lag)  # Log transformation for GHG paths
        log_ghg_level = np.log1p(
            ghg_level_train_in.detach().cpu().numpy() / (self.w0 * portret_train_in.detach().cpu().numpy()))
        standardized_covpaths = StandardScaler().fit_transform(
            self.upper_triangle(covpaths_train_lag))  # Standardize covariance
        cpath_train[:, 0] = portret_train_in.detach().cpu().numpy()
        cpath_train[:, 1] = (tt_train - 1) / self.nsteps
        cpath_train[:, 2] = log_ghg_level
        cpath_train[:, 3:14] = log_ghg_path
        cpath_train[:, 14:] = standardized_covpaths
        return torch.tensor(cpath_train, dtype=torch.float32).to(self.device)

    def train_model(self):
        objective_value_train = torch.Tensor([0.0])

        for nn_train in range(self.niter):
            # Generate minibatch of random samples
            batchrowid = np.array(range(self.minibatchsize)) + nn_train * self.minibatchsize
            returnpaths_train = self.simullogreturns_all_IS[batchrowid,]
            ghg_path_train = self.ghg_paths_all_IS[batchrowid,]
            covpaths_train = self.simulcovariance_IS[batchrowid,]

            portret_train = torch.ones(self.minibatchsize).to(self.device)
            ghg_level_train = torch.zeros(self.minibatchsize).to(self.device)
            cpath_train = np.zeros((self.minibatchsize, self.input_size))

            self.optimizer.zero_grad()  # Clear accumulated gradients

            for tt_train in range(1, self.nsteps):
                transformed_input = self.prepare_inputs(cpath_train, covpaths_train[:, :, :, tt_train - 1],
                                                        ghg_path_train[:, tt_train - 1, :], portret_train,
                                                        ghg_level_train, tt_train)
                output = self.policy_network(transformed_input)
                portret_old_train = portret_train.clone()
                retvec_train = np.reshape(returnpaths_train[:, tt_train], (self.minibatchsize, self.NRiskyAssets))
                portret_train = (portret_train *
                                 torch.sum(output * torch.tensor(retvec_train, dtype=torch.float32).to(self.device),
                                           dim=1))
                ghgvec_train = np.reshape(ghg_path_train[:, tt_train], (self.minibatchsize, self.NRiskyAssets))
                ghg_level_train = ghg_level_train + self.w0 * portret_old_train * torch.sum(
                    output * torch.tensor(ghgvec_train, dtype=torch.float32).to(self.device), dim=1)

            objective_value_train = objective_function(self.w0 * portret_train, ghg_level_train, rho_coef=self.rho,
                                                       coef_ghg=self.gamma)

            objective_value_train.backward()
            self.optimizer.step()

        return objective_value_train

    def evaluate_model(self, on: Literal['OOS', 'IS'] = 'OOS'):
        if on not in ['OOS', 'IS']:
            raise ValueError("Parameter 'on' must be either 'OOS' or 'IS'")
        self.policy_network.eval()
        with torch.no_grad():
            numpaths = self.simullogreturns_all_OOS.shape[0]
            returnpaths_eval = self.simullogreturns_all_OOS
            ghg_path_eval = self.ghg_paths_all_OOS
            covpaths_eval = self.simulcovariance_OOS
            if on == 'IS':
                numpaths = self.simullogreturns_all_IS.shape[0]
                returnpaths_eval = self.simullogreturns_all_IS
                ghg_path_eval = self.ghg_paths_all_IS
                covpaths_eval = self.simulcovariance_IS

            portret_eval = torch.ones(numpaths).to(self.device)
            ghg_level_eval = torch.zeros(numpaths).to(self.device)
            cpath_eval = np.zeros((numpaths, self.input_size))

            for tt_eval in range(1, self.nsteps):
                transformed_input = self.prepare_inputs(cpath_eval, covpaths_eval[:, :, :, tt_eval - 1],
                                                        ghg_path_eval[:, tt_eval - 1, :], portret_eval,
                                                        ghg_level_eval, tt_eval)
                output = self.policy_network(transformed_input)
                portret_old_eval = portret_eval.clone()
                retvec_eval = np.reshape(returnpaths_eval[:, tt_eval], (numpaths, self.NRiskyAssets))
                portret_eval = portret_eval * torch.sum(output *
                                                        torch.tensor(retvec_eval, dtype=torch.float32).to(self.device),
                                                        dim=1)
                ghgvec_eval = np.reshape(ghg_path_eval[:, tt_eval], (numpaths, self.NRiskyAssets))
                ghg_level_eval = ghg_level_eval + self.w0 * portret_old_eval * torch.sum(
                    output * torch.tensor(ghgvec_eval, dtype=torch.float32).to(self.device), dim=1)

            objective_value_eval = objective_function(self.w0 * portret_eval, ghg_level_eval, rho_coef=self.rho,
                                                      coef_ghg=self.gamma)

        output_list = {'objective_value': objective_value_eval, 'portret': portret_eval, 'ghg_level': ghg_level_eval}
        return output_list

    def train_evaluate_model(self, ISeval=True):
        print('Training started ... ')
        es = EralyStopping(patience_es=self.patience, min_delta=0.0, path=self.network_path)
        # storing estimates of performance after each epoch
        PerfvecIS_treval = torch.tensor([])
        PerfvecOOS_treval = torch.tensor([])
        PerfvecIS_treval = PerfvecIS_treval.to(self.device)
        PerfvecOOS_treval = PerfvecOOS_treval.to(self.device)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, min_lr=0.001,
                                                         patience=5)

        for ee in range(self.nepochs):
            # Train model
            self.policy_network.train(True)
            objective_value_train = self.train_model()
            objective_value_train = objective_value_train.to(self.device)
            PerfvecIS_treval = torch.cat((PerfvecIS_treval, objective_value_train.unsqueeze(0)), dim=0)

            if ISeval:
                # Evaluate model
                objective_value_eval = self.evaluate_model(on='OOS')
                objective_value_tensor = torch.tensor([objective_value_eval['objective_value'].item()])
                PerfvecOOS_treval = torch.cat((PerfvecOOS_treval, objective_value_tensor), dim=0)

            # Check for early stopping
            es(objective_value_train.item(), self.policy_network)
            if es.earlystop:
                print(
                    f"Early stopping at epoch {ee + 1} with objective value {objective_value_train.item()} and"
                    f" best value {es.best_objective_value} after {es.wait} epochs")
                break
            scheduler.step(PerfvecIS_treval[-1])

            if (ee + 1) % 10 == 0:
                print(f"Epoch {ee + 1}/{self.nepochs} - Final objective value: {objective_value_train.item()}"
                      f" and learning rate is {self.optimizer.param_groups[0]['lr']}")
            if (ee + 1) == self.nepochs:
                print(f"Terminated at epoch {ee + 1} with best objective value {es.best_objective_value}")
        return PerfvecIS_treval, PerfvecOOS_treval
