import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np

class BNN(nn.Module):
    """NCI prototype → Laplace approximation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    def model(self, x_data, y_data=None):
        # Priors for weights
        fc1_w_prior = dist.Normal(0., 1.).expand([self.hidden_dim, self.input_dim]).to_event(2)
        fc1_b_prior = dist.Normal(0., 1.).expand([self.hidden_dim]).to_event(1)
        
        fc2_w_prior = dist.Normal(0., 1.).expand([self.hidden_dim, self.hidden_dim]).to_event(2)
        fc2_b_prior = dist.Normal(0., 1.).expand([self.hidden_dim]).to_event(1)
        
        fc3_w_prior = dist.Normal(0., 1.).expand([self.output_dim, self.hidden_dim]).to_event(2)
        fc3_b_prior = dist.Normal(0., 1.).expand([self.output_dim]).to_event(1)
        
        priors = {
            'fc1.weight': fc1_w_prior,
            'fc1.bias': fc1_b_prior,
            'fc2.weight': fc2_w_prior, 
            'fc2.bias': fc2_b_prior,
            'fc3.weight': fc3_w_prior,
            'fc3.bias': fc3_b_prior
        }
        
        # Lift module parameters to random variables
        lifted_module = pyro.random_module("module", self, priors)
        lifted_reg_model = lifted_module()
        
        with pyro.plate("data_processing", x_data.shape[0]):
            prediction_mean = lifted_reg_model(x_data).squeeze(-1)
            pyro.sample("obs", dist.Normal(prediction_mean, 0.1), obs=y_data)
            
        return prediction_mean
    
    def guide(self, x_data, y_data=None):
        # Variational distributions (mean-field approximation)
        fc1_w_mu_param = pyro.param("fc1_w_mu", torch.randn([self.hidden_dim, self.input_dim]))
        fc1_w_sigma_param = pyro.param("fc1_w_sigma", torch.randn([self.hidden_dim, self.input_dim]), 
                                     constraint=dist.constraints.positive)
        fc1_w_dist = dist.Normal(fc1_w_mu_param, fc1_w_sigma_param).to_event(2)
        
        fc1_b_mu_param = pyro.param("fc1_b_mu", torch.randn([self.hidden_dim]))
        fc1_b_sigma_param = pyro.param("fc1_b_sigma", torch.randn([self.hidden_dim]), 
                                     constraint=dist.constraints.positive)
        fc1_b_dist = dist.Normal(fc1_b_mu_param, fc1_b_sigma_param).to_event(1)
        
        fc2_w_mu_param = pyro.param("fc2_w_mu", torch.randn([self.hidden_dim, self.hidden_dim]))
        fc2_w_sigma_param = pyro.param("fc2_w_sigma", torch.randn([self.hidden_dim, self.hidden_dim]), 
                                     constraint=dist.constraints.positive)
        fc2_w_dist = dist.Normal(fc2_w_mu_param, fc2_w_sigma_param).to_event(2)
        
        fc2_b_mu_param = pyro.param("fc2_b_mu", torch.randn([self.hidden_dim]))
        fc2_b_sigma_param = pyro.param("fc2_b_sigma", torch.randn([self.hidden_dim]), 
                                     constraint=dist.constraints.positive)
        fc2_b_dist = dist.Normal(fc2_b_mu_param, fc2_b_sigma_param).to_event(1)
        
        fc3_w_mu_param = pyro.param("fc3_w_mu", torch.randn([self.output_dim, self.hidden_dim]))
        fc3_w_sigma_param = pyro.param("fc3_w_sigma", torch.randn([self.output_dim, self.hidden_dim]), 
                                     constraint=dist.constraints.positive)
        fc3_w_dist = dist.Normal(fc3_w_mu_param, fc3_w_sigma_param).to_event(2)
        
        fc3_b_mu_param = pyro.param("fc3_b_mu", torch.randn([self.output_dim]))
        fc3_b_sigma_param = pyro.param("fc3_b_sigma", torch.randn([self.output_dim]), 
                                     constraint=dist.constraints.positive)
        fc3_b_dist = dist.Normal(fc3_b_mu_param, fc3_b_sigma_param).to_event(1)
        
        variational_dist = {
            'fc1.weight': fc1_w_dist,
            'fc1.bias': fc1_b_dist,
            'fc2.weight': fc2_w_dist,
            'fc2.bias': fc2_b_dist,
            'fc3.weight': fc3_w_dist,
            'fc3.bias': fc3_b_dist
        }
        
        lifted_module = pyro.random_module("module", self, variational_dist)
        return lifted_module()
    
    def train(self, x_data, y_data, num_epochs: int = 1000, lr: float = 0.01):
        """Train the BNN with SVI"""
        optimizer = Adam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)
        
        losses = []
        for epoch in range(num_epochs):
            loss = svi.step(x_tensor, y_tensor)
            losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                
        return losses
    
    def predict(self, x_data, num_samples: int = 100):
        """Make predictions with uncertainty"""
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        
        predictions = []
        for _ in range(num_samples):
            sampled_model = self.guide(x_tensor)
            pred = sampled_model(x_tensor).detach().numpy()
            predictions.append(pred)
            
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
