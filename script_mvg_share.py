import math
import pickle
import time
import torch
import torch.optim as optim
from Mean_Var_GHG import PolicyNetwork, OptimizationModel

# Open the pickle file
with open('GHG_PATH25K_seed1245.pkl', 'rb') as f:
    loaded_variables_ghg = pickle.load(f)
with open('DCC_EGARCH_PATH25K_standard_seed1245_new.pkl', 'rb') as f:
    loaded_variables_dcc = pickle.load(f)

simullogreturnsIS = loaded_variables_dcc['simullogreturnsIS']
simulvolatilitiesIS = loaded_variables_dcc['simulvolatilitiesIS']
simulcovarianceIS = loaded_variables_dcc['simulcovarianceIS']
simullogreturnsOOS = loaded_variables_dcc['simullogreturnsOOS']
simulvolatilitiesOOS = loaded_variables_dcc['simulvolatilitiesOOS']
simulcovarianceOOS = loaded_variables_dcc['simulcovarianceOOS']
ghg_paths_IS_monthly = loaded_variables_ghg['ghg_paths_IS_monthly']
ghg_paths_OOS_monthly = loaded_variables_ghg['ghg_paths_OOS_monthly']
NRiskyAssets = loaded_variables_ghg['NRiskyAssets']
sector_names = loaded_variables_ghg['sector_names']


del loaded_variables_ghg
del loaded_variables_dcc

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

npaths = 25000
nsteps = 60
init_wealth = 100

input_size = 3 + int(NRiskyAssets * (NRiskyAssets + 1) / 2) + NRiskyAssets
hidden_size = [4]
output_size = NRiskyAssets

# learning parameters
nepochs = 5000  # number of epochs during training
minibatchsize = 128  # number of paths fed into the minibatch for each parameter update
learingrate = 0.01
if len(hidden_size) > 2:
    learingrate = 0.01  # parameter related to learning speed  0.003  0.001 0.0001

# Training loop with early stopping
patience = 200  # Number of epochs to wait before early stopping

niter = math.floor(npaths / minibatchsize)  # number of parameter updates per epoch

dataIS = {
    'simullogreturns_all': simullogreturnsIS + 1,
    'simulcovariance': simulcovarianceIS,
    'ghg_paths_all': ghg_paths_IS_monthly
}

dataOOS = {
    'simullogreturns_all': simullogreturnsOOS + 1,
    'simulcovariance': simulcovarianceOOS,
    'ghg_paths_all': ghg_paths_OOS_monthly
}

rho = 0.02
gamma = 0.1

torch.manual_seed(2024)
# Create an instance of the policy network
netpth = f'policy_network_deep.pth'
policy_network = PolicyNetwork(input_size, hidden_size, output_size).to(device)
optimizer = optim.Adam(policy_network.parameters(), lr=learingrate, maximize=True)
opmodel = OptimizationModel(policy_network=policy_network, optimizer=optimizer, n_epochs=nepochs,
                            minibatchsize=minibatchsize, niter=niter, input_size=input_size,
                            rho=rho, gamma=gamma, network_path=netpth, patience=patience,
                            data_in_sample=dataIS, data_out_of_sample=dataOOS,
                            wealth_init=init_wealth, NRiskyAssets=NRiskyAssets, n_steps=nsteps, device_p=device)
start_time = time.time()
PerfvecIS, PerfvecOOS = opmodel.train_evaluate_model(ISeval=False)
end_time = time.time()
train_time = (end_time - start_time) / 60

policy_network.load_state_dict(torch.load(netpth))
objective_value_IS = opmodel.evaluate_model(on='IS')
objective_value_OOS = opmodel.evaluate_model(on='OOS')

print(f'trainning time is {train_time}')
print(f"In sample objective value: {objective_value_IS['objective_value']}"
      f" and Out of sample objective value: {objective_value_OOS['objective_value']}")

pass
