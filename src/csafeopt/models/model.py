from typing import Optional

import gpytorch
import torch
from botorch.models import ModelListGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean
from torch import Tensor

import csafeopt
from csafeopt.tools.data import Data


class ModelGenerator:
    def __init__(
        self,
        dim_model: int,
        dim_obs: int,
        likelihood_noise: Tensor,
        lenghtscale: Tensor,
        normalize_input: bool = False,
        normalize_output: bool = False,
        domain_start: Optional[Tensor] = None,
        domain_end: Optional[Tensor] = None,
        state_dict: Optional[dict] = None,
        constraint_lengthscale: Optional[Tensor] = None,
        outputscale: Optional[float] = None,
    ):
        if normalize_input and (domain_start is None or domain_end is None):
            raise Exception("If normalize_input is True domain bounds have to be set")

        self.domain_start = domain_start
        self.domain_end = domain_end
        self.dim_input = dim_model
        self.dim_obs = dim_obs
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.likelihood_noise = likelihood_noise
        self.lengthscale = lenghtscale
        self.constraint_lengthscale = constraint_lengthscale
        self.outputscale = outputscale
        self.state_dict = state_dict
        self.fitted_parameters = {}

    def remember_fitted_parameters(self, model: ModelListGP):
        # Preserve learned hyperparameters, not data-dependent normalization
        # buffers or fixed means, which must be rebuilt for the current data.
        self.fitted_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def generate(self, data: Data) -> ModelListGP:
        if data.train_x is None or data.train_y is None:
            raise Exception("Data can not be emtpy")

        if self.normalize_input and self.domain_start is not None and self.domain_end is not None:
            input_transform = Normalize(self.dim_input, bounds=torch.vstack([self.domain_start, self.domain_end]))
        else:
            input_transform = None

        models = []

        for i in range(self.dim_obs):
            mean_module = ConstantMean()
            outcome_transform = None

            # The demos initialize noise at 1e-4, exactly the default lower
            # bound. That gives a raw noise parameter of -inf and breaks fitting.
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
            )
            likelihood.noise = self.likelihood_noise
            likelihood.to(csafeopt.device)

            # TODO: how to update outcome_transform with condition on observation
            if self.normalize_output:
                outcome_transform = Standardize(m=1)
                outcome_transform.train()
                outcome_transform(data.train_y[:, i].reshape(-1, 1))[0]
                outcome_transform.eval()
                mean_module.constant.requires_grad_(False)
                if i > 0:
                    mean_module.constant = outcome_transform(torch.zeros(1, 1))[0]
                else:
                    # mean_module operates in the standardized space Standardize maps
                    # training data into, where the mean is 0 by construction -- not
                    # outcome_transform.means (the raw-space mean), which is what large
                    # reward ranges away from 0 previously biased this prior mean by.
                    mean_module.constant = torch.zeros_like(outcome_transform.means[0])

            covar_module = ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=self.dim_input))
            lengthscale = self.constraint_lengthscale if i > 0 and self.constraint_lengthscale is not None else self.lengthscale
            covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
            if self.outputscale is not None:
                covar_module.outputscale = self.outputscale

            models.append(
                SingleTaskGP(
                    data.train_x,
                    data.train_y[:, i].reshape(-1, 1),
                    likelihood=likelihood,
                    covar_module=covar_module,
                    mean_module=mean_module,
                    outcome_transform=outcome_transform,
                    input_transform=input_transform,
                ).to(csafeopt.device)
            )

        model = ModelListGP(*models)

        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.fitted_parameters:
                    parameter.copy_(self.fitted_parameters[name])

        return model
