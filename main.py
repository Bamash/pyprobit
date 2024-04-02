import pyjags
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm, mvn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import math
from itertools import product

model_code_template = model_code_template = """ 
model {{
  # Declare likelihood for Y, relationship between Y and Y_s
  for (i in 1:n) {{
    for (q in 1:{n_outputs}) {{  # Extend loop for the third outcome
      Y[i, q] ~ dinterval(Z[i, q], 0)
      mu[i, q] <- X[i,] %*% Beta[, q]
    }}
    Z[i, 1:{n_outputs}] ~ dmnorm.vcov(mu[i, ], prec[1:{n_outputs}, 1:{n_outputs}])  # Extend to accommodate the third outcome
  }}

  # Prior on Betas
  for (q in 1:{n_outputs}) {{  # Extend loop for the third outcome
    Beta[1:P, q] ~ dmnorm(b_0, B_0)
  }}

  # Prior on covariance matrix
  prec[1:{n_outputs}, 1:{n_outputs}] <- cov[, ]
  {covariance_block}
  # Flat priors on all parameters which could, of course, be made more informative.
  for (i in 1:{n_outputs}) {{
    sigma[i] = 1  # Extend to accommodate the third outcome
  }}
  
  rho12 ~ dunif(-1, 1)  # Prior for rho12
  rho13 ~ dunif(-1, 1)  # Prior for rho13
  rho23 ~ dunif(-1, 1)  # Prior for rho23
}}
"""


class Probit_Model:
    def __init__(self, X, Y, model_code = model_code_template):
        self.X = X
        self.Y = Y
        self.model_code = model_code

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2)

        self.n_params = self.X.shape[1]
        self.n_outputs = self.Y.shape[1]
        self.data = {
            "n": self.X_train.shape[0],
            "P": self.n_params + 1,  # Number of inputs + 1 (for intercept)
            "Y": np.column_stack([self.y_train['Y{}'.format(i)] for i in range(1, self.n_outputs + 1)]),
            "X": np.column_stack([np.ones(self.X_train.shape[0])] + [self.X_train['X{}'.format(i)] for i in range(1, self.n_params + 1)]),
            "b_0": np.zeros(self.n_params + 1),  # Number of inputs + 1 (for intercept)
            "B_0": np.diag(np.repeat(0.1, self.n_params + 1))  # Precision (Number of inputs + 1)
        }
        self.rho_params = [f'rho{i}{j}' for i in range(1, self.n_outputs + 1) for j in range(i + 1, self.n_outputs + 1)]
        
        covariance_block = ""
        for i in range(1, self.n_params + 1):
            for j in range(1, self.n_params + 1):
                if i == j:
                    covariance_block += f"  cov[{i}, {j}] <- sigma[{i}] * sigma[{j}]\n"
                else:
                    covariance_block += f"  cov[{i}, {j}] <- sigma[{i}] * sigma[{j}] * rho{min(i,j)}{max(i,j)}\n"
        
        
        # Fill in the placeholders in the Python code template
        self.model_code = self.model_code.format(n_outputs=self.n_outputs,
            covariance_block=covariance_block
        )

    def initialise_model(self, chains):
        self.initial_values = {"Z": test.y_train[[f'Y{i}' for i in range(1, test.n_outputs + 1)]].values}
        for i in self.rho_params:
            self.initial_values[i] = 0
        # Compile the model
        self.model = pyjags.Model(code=self.model_code, data=self.data, chains = chains, init=self.initial_values, threads = chains)

    def fit(self, iterations = 10000):
        self.samples = self.model.sample(vars=['Beta'] + self.rho_params, iterations=iterations)

    def _extract_posterior_means(self, burn_in=5000):
        # Extract correlation parameters
        rho_post_means = {f'rho{i}{j}_post_mean': np.mean(self.samples[self.rho_params[(i - 1) * (self.n_outputs - i) + (j - i - 1)]][0][burn_in:]) 
                          for i in range(1, self.n_outputs) for j in range(i + 1, self.n_outputs + 1)}
    
        # Extract regression coefficients
        beta = self.samples['Beta'][:, :, burn_in:, :]
        beta_post_means = {f'beta{q}_post_mean': np.mean(beta[:, q - 1, :, 0], axis=1) for q in range(1, self.n_outputs + 1)}
    
        return {**rho_post_means, **beta_post_means}

    def predict(self):
        parameters_dic = self._extract_posterior_means()
