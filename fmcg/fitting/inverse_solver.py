import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from scipy.optimize import basinhopping, linear_sum_assignment
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import trange
import pytorch_optimizer as optim2

from .basis import *
from ..utils.utils import Logger, TqdmLoggingHandler
from ._inverse_utils import *
from .field_model import ForwardModel

logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.hasHandlers():
    logger.addHandler(TqdmLoggingHandler())

class InverseSolver:

    def __init__(
        self,
        r_sensors,
        method="adam",
        bnds=None,
        basis=None,
        axis_mask=None,
        num_dipoles=2,
        scaling=None,
        device="cpu",
        whitening_matrix=None,
        verbose=True,
        **kwargs,
    ):

        # initialize bounds
        if bnds is not None:
            for k in bnds.keys():
                if not isinstance(bnds[k], torch.Tensor):
                    bnds[k] = torch.tensor(bnds[k], dtype=torch.float32, device=device)
                bnds[k] = bnds[k].unsqueeze(0)
        self.bnds = bnds

        self.pivot_index = None
        self.num_dipoles = num_dipoles
        self.dof_per_dipole = 3
        self.device = device
        self.field_scaling = None
        self.method = method
        if method in ["basinhopping"]:
            self.parameter_format = "vector"
        elif method in ["adam", "nested", "twostep", "lamb"]:
            self.parameter_format = "groups"
        else:
            raise NotImplementedError(f"Method {method} not implemented!")

        # initialize basis
        self.basis = {}
        if basis is not None:
            for k in basis.keys():
                if basis[k]["basis"] == "bspline":
                    assert "n_time" in basis[k]
                    n_time = basis[k]["n_time"]
                    n_knots = basis[k]["n_knots"] if "n_knots" in basis[k] else 10
                    degree = basis[k]["degree"] if "degree" in basis[k] else 3
                    self.basis[k] = BSpline(n_knots, n_time, degree=degree, device=self.device)
                elif basis[k]["basis"] == "identity":
                    assert "n_time" in basis[k]
                    self.basis[k] = IdentityBasis(basis[k]["n_time"], device=self.device)
                elif basis[k]["basis"] == "tanh_time_constant":
                    assert "n_time" in basis[k]
                    assert "bnds" in basis[k]
                    self.basis[k] = TanhBasis(basis[k]["bnds"], basis[k]["n_time"], device=self.device)
                elif basis[k]["basis"] == "sigmoid_time_constant":
                    assert "n_time" in basis[k]
                    assert "bnds" in basis[k]
                    self.basis[k] = SigmoidBasis(basis[k]["bnds"], basis[k]["n_time"], device=self.device)
                elif basis[k]["basis"] == "exponential":
                    base = basis[k].get("base", "10")
                    self.basis[k] = ExponentialBasis(
                        basis=base, n_time=basis[k].get("n_time", None), device=self.device
                    )
                elif basis[k]["basis"] == "wavelet":
                    assert "n_time" in basis[k]
                    self.basis[k] = WaveletBasis(n_time=basis[k].get("n_time", None), wavelet=basis[k].get("wavelet", "db4"), device=self.device, level=basis[k].get("level", None))

        # initialize axis masking for removing non-recorded field components
        if axis_mask is None:
            axis_mask = torch.ones((16, 3), dtype=torch.bool, device=self.device)
        if not isinstance(axis_mask, torch.Tensor):
            axis_mask = torch.tensor(axis_mask, dtype=torch.bool, device=self.device)
        axis_mask = axis_mask.to(torch.bool)
        self.axis_mask = axis_mask
        #assert axis_mask.shape == (16, 3)

        # initialize forward model
        if not isinstance(r_sensors, torch.Tensor):
            r_sensors = torch.tensor(r_sensors, dtype=torch.float32, device=device)
        self.forward_model = ForwardModel(r_sensors, device=device, axis_mask=axis_mask)

        if whitening_matrix is not None and isinstance(whitening_matrix, (list, np.ndarray)):
            whitening_matrix = torch.tensor(whitening_matrix, dtype=torch.float32, device=device)
        self.whitening_matrix = whitening_matrix

        # create labels for parameters
        self.parameter_labels = [
            f"{label}_{i+1}" for i in range(num_dipoles) for label in [r"$r_x$", r"$r_y$", r"$r_z$"]
        ]

        # initialize scaling
        self.scaling_init = {}
        if scaling is not None:
            for k in scaling.keys():
                if not isinstance(scaling[k], torch.Tensor):
                    scaling[k] = torch.tensor(scaling[k], dtype=torch.float32, device=self.device)
                self.scaling_init[k] = scaling[k]
                if self.scaling_init[k].ndim == 1:
                    self.scaling_init[k] = self.scaling_init[k].unsqueeze(-1).unsqueeze(0)

        self.scaling = {k: v.clone() for k, v in self.scaling_init.items()}

        # initialize callback logger
        self.callback = Logger(parameter_labels=self.parameter_labels)

        self.verbose = verbose

    def _get_parameter_structure(self):
        """Get the current parameter structure based on basis functions."""
        return {
            "m_has_basis": "m" in self.basis,
            "r_has_basis": "r" in self.basis,
            "num_dipoles": self.num_dipoles
        }
    
    def _create_parameter_groups(self, parameters):
        """Create parameter groups for optimization based on basis function configuration."""
        if not isinstance(parameters, list):
            return torch.nn.Parameter(parameters.detach().clone().requires_grad_(True))
        
        m_tensor, r_tensor = parameters[0], parameters[1]
        structure = self._get_parameter_structure()
        
        # Create magnetic moment parameters
        if structure["m_has_basis"]:
            m_params = [torch.nn.Parameter(m_tensor.detach().clone().requires_grad_(True))]
        else:
            m_params = self._create_separate_dipole_params(m_tensor)
        
        # Create position parameters  
        if structure["r_has_basis"]:
            r_params = [torch.nn.Parameter(r_tensor.detach().clone().requires_grad_(True))]
        else:
            r_params = self._create_separate_dipole_params(r_tensor)
        
        return m_params + r_params
    
    def _create_separate_dipole_params(self, tensor):
        """Create separate parameter tensors for each dipole."""
        viewd = tensor.view(tensor.shape[0], self.num_dipoles, self.dof_per_dipole)
        return [torch.nn.Parameter(viewd[:, i, :].detach().clone().requires_grad_(True)) 
                for i in range(self.num_dipoles)]
    
    def _setup_optimizer_groups(self, parameters, lr_config):
        """Setup optimizer parameter groups with appropriate learning rates."""
        structure = self._get_parameter_structure()
        lr_groups = lr_config.get("lr_groups", [])
        default_lr = lr_config.get("start", 1e-1) #TODO
        default_pos_lr = lr_config.get("pos_start", default_lr)
        
        param_groups = []
        lr_idx = 0
        
        # Add magnetic moment groups
        if structure["m_has_basis"]:
            lr_val = lr_groups[lr_idx] if lr_idx < len(lr_groups) else default_lr
            param_groups.append({"params": [parameters[0]], "lr": lr_val})
            lr_idx += 1
        else:
            for dipole_idx in range(structure["num_dipoles"]):
                lr_val = lr_groups[lr_idx] if lr_idx < len(lr_groups) else default_lr
                param_groups.append({"params": [parameters[dipole_idx]], "lr": lr_val})
                lr_idx += 1
        
        # Add position groups
        pos_start_idx = structure["num_dipoles"] if not structure["m_has_basis"] else 1
        if structure["r_has_basis"]:
            lr_val = lr_groups[lr_idx] if lr_idx < len(lr_groups) else default_pos_lr
            param_groups.append({"params": [parameters[pos_start_idx]], "lr": lr_val})
        else:
            for dipole_idx in range(structure["num_dipoles"]):
                lr_val = lr_groups[lr_idx] if lr_idx < len(lr_groups) else default_pos_lr
                param_groups.append({"params": [parameters[pos_start_idx + dipole_idx]], "lr": lr_val})
                lr_idx += 1
        
        return param_groups
    
    def _get_learning_rates_dict(self, optimizer):
        """Extract learning rates from optimizer parameter groups into a dictionary."""
        structure = self._get_parameter_structure()
        lr_dict = {}
        group_idx = 0
        
        # Magnetic moment learning rates
        if structure["m_has_basis"]:
            lr_dict["lr_m"] = optimizer.param_groups[group_idx]["lr"]
            group_idx += 1
        else:
            for dipole_idx in range(structure["num_dipoles"]):
                lr_dict[f"lr_m_{dipole_idx}"] = optimizer.param_groups[group_idx]["lr"]
                group_idx += 1
        
        # Position learning rates
        if structure["r_has_basis"]:
            lr_dict["lr_r"] = optimizer.param_groups[group_idx]["lr"]
        else:
            for dipole_idx in range(structure["num_dipoles"]):
                lr_dict[f"lr_r_{dipole_idx}"] = optimizer.param_groups[group_idx]["lr"]
                group_idx += 1
        
        return lr_dict
        
    def _calculate_group_losses(self, parameters, field_true):
        """Calculate loss components for each parameter group (simplified version)."""
        # For now, return the same loss for all groups
        # In the future, this could compute individual contributions
        return [self.loss_value] * len(parameters) if isinstance(parameters, list) else [self.loss_value]
        
    def _setup_schedulers(self, optimizer, lr_config):
        """Setup individual schedulers for each parameter group with tunable patience."""
        warmup = lr_config.get("warmup", 50)
        niter = lr_config.get("niter", 200)
        end_lr = lr_config.get("end", 1e-4)
        
        # Get individual patience values if provided
        patience_values = lr_config.get("patience_groups", [])
        default_patience = patience_values[0] if patience_values else 20
        
        # Debug output
        logger.debug(f"Using individual patience values: {patience_values}")
        
        # Create both warmup and plateau schedulers for each group
        warmup_schedulers = []
        plateau_schedulers = []
        
        for idx, group in enumerate(optimizer.param_groups):
            # Get patience for this group
            patience = patience_values[idx] if idx < len(patience_values) else default_patience
            logger.debug(f"Group {idx} ({list(group['params'][0].shape)}): patience={patience}")

            # Create warmup scheduler using cosine annealing for smooth ramp-up
            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=lambda epoch, idx=idx: (1 - np.cos(np.pi * (epoch + 1) / max(warmup, 1))) / 2,
                last_epoch=-1
            )
            warmup_schedulers.append(warmup_scheduler)
            
            # Create plateau scheduler with individual patience
            plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.9,
                patience=patience, 
                threshold=1e-5,
            )
            plateau_schedulers.append(plateau_scheduler)
            
            logger.debug(f"Scheduler {idx}: Warmup for {warmup} steps, Plateau with patience {patience}")
        
        return {
            "warmup": warmup_schedulers,
            "plateau": plateau_schedulers,
            "warmup_steps": warmup
        }

    def parametervector2parameters(self, parameters, apply_basis=True):
        """
        Converts a parameter vector into separate parameter arrays for dipoles. This is useful for
        optimization tasks where the parameters are represented as a single vector (e.g., BasinHopping) or where
        they can be represented as separate arrays / groups (e.g., Adam).

        Args:
            parameters (torch.Tensor): The input parameter vector.

        Returns:
            tuple: A tuple containing:
                - m_dipoles (torch.Tensor): The magnetic dipole parameters.
                - r_dipoles (torch.Tensor): The position dipole parameters.

        """

        if self.parameter_format == "vector":
            m_dipoles = parameters[: self.pivot_index].view(-1, self.num_dipoles * self.dof_per_dipole)
            r_dipoles = parameters[self.pivot_index :].view(-1, self.num_dipoles * self.dof_per_dipole)
        else:
            # Get parameter structure and convert accordingly
            structure = self._get_parameter_structure()
            
            if structure["m_has_basis"] and structure["r_has_basis"]:
                # Both have basis: parameters = [m_coeffs, r_coeffs]
                m_dipoles, r_dipoles = parameters[0], parameters[1]
            elif structure["m_has_basis"] and not structure["r_has_basis"]:
                # m has basis, r separated: parameters = [m_coeffs, r_dipole_0, r_dipole_1, ...]
                m_dipoles = parameters[0]
                r_dipoles = torch.stack(parameters[1:], dim=1).view(parameters[1].shape[0], -1)
            elif not structure["m_has_basis"] and structure["r_has_basis"]:
                # m separated, r has basis: parameters = [m_dipole_0, m_dipole_1, ..., r_coeffs]
                m_dipoles = torch.stack(parameters[:-1], dim=1).view(parameters[0].shape[0], -1)
                r_dipoles = parameters[-1]
            else:
                # Neither has basis: parameters = [m_dipole_0, m_dipole_1, ..., r_dipole_0, r_dipole_1, ...]
                mid_point = structure["num_dipoles"]
                m_dipoles = torch.stack(parameters[:mid_point], dim=1).view(parameters[0].shape[0], -1)
                r_dipoles = torch.stack(parameters[mid_point:], dim=1).view(parameters[0].shape[0], -1)

        if apply_basis:
            # Apply basis functions to coefficients
            if "m" in self.basis:
                m_dipoles = self.basis["m"].forward(m_dipoles)
            if "r" in self.basis:
                r_dipoles = self.basis["r"].forward(r_dipoles)

        m_dipoles = m_dipoles.view(-1, self.num_dipoles, self.dof_per_dipole)
        r_dipoles = r_dipoles.view(-1, self.num_dipoles, self.dof_per_dipole)

        if "m" in self.scaling:
            m_dipoles = m_dipoles * self.scaling["m"]
        if "r" in self.scaling:
            r_dipoles = r_dipoles * self.scaling["r"]

        return m_dipoles, r_dipoles

    def parameters2parametervector(self, m_dipoles, r_dipoles, as_numpy=False, determine_scaling=False):
        """
        Converts dipole parameters into a parameter vector or a list of parameters.

        Args:
            m_dipoles (torch.Tensor or array-like): Magnetic dipole moments.
            r_dipoles (torch.Tensor or array-like): Position vectors of the dipoles.
            as_numpy (bool, optional): If True, returns the parameters as numpy arrays. Defaults to False.
            determine_scaling (bool, optional): [legacy] If True, determines and applies scaling factors to the parameters. Defaults to False.

        Returns:
            torch.Tensor or list: If `parameter_format` is "vector", returns a concatenated parameter vector.
                                  Otherwise, returns a list of magnetic and position dipole parameters.
                                  If `as_numpy` is True, returns numpy arrays instead of torch tensors.
        """
        if not isinstance(m_dipoles, torch.Tensor):
            m_dipoles = torch.tensor(m_dipoles, dtype=torch.float32, device=self.device)
        if not isinstance(r_dipoles, torch.Tensor):
            r_dipoles = torch.tensor(r_dipoles, dtype=torch.float32, device=self.device)

        if determine_scaling:
            cat = torch.cat((m_dipoles, r_dipoles), dim=1)
            scale = torch.linalg.vector_norm(cat, ord=1, dim=(0, 2)) / torch.linalg.vector_norm(cat, ord=1)
            self.scaling["m"] = scale[: m_dipoles.shape[1]].unsqueeze(-1)
            self.scaling["r"] = scale[m_dipoles.shape[1] :].unsqueeze(-1)

        if "m" in self.scaling:
            m_dipoles = m_dipoles / self.scaling["m"]
        if "r" in self.scaling:
            r_dipoles = r_dipoles / self.scaling["r"]

        if "m" in self.basis:
            m_dipoles = self.basis["m"].fit_coefficients(m_dipoles.view(-1, self.num_dipoles * self.dof_per_dipole))
        if "r" in self.basis:
            r_dipoles = self.basis["r"].fit_coefficients(r_dipoles.view(-1, self.num_dipoles * self.dof_per_dipole))

        if self.parameter_format == "vector":
            parameters = torch.cat((m_dipoles.view(-1), r_dipoles.view(-1)), dim=-1)
            self.pivot_index = len(m_dipoles.view(-1))
            if as_numpy:
                return parameters.detach().cpu().numpy()
        else:
            if as_numpy:
                return [m_dipoles.detach().cpu().numpy(), r_dipoles.detach().cpu().numpy()]
            return [m_dipoles, r_dipoles]

    def initialize_parameters(self, r:torch.Tensor, field_true:torch.Tensor, m=None, method="pseudo_inv", field_scaling=1e0, determine_scaling=False, update_m_scaling=False, rcond=1e-4):
        """
        Initialize the parameters for the model.

        Args:
            r (torch.Tensor): Input tensor representing the positions.
            field_true (torch.Tensor): The measured field values.
            method (str, optional): The method to use for initialization. Default is "pseudo_inv", which uses the
            pseudo-inverse to estimates initial parameters.
            field_scaling (float, optional): Scaling factor for the field. Default is 1e3.
            update_m_scaling (bool, optional): Whether to update the scaling for the magnetic dipole moment. Default is
            False.
            determine_scaling (bool, optional): Whether to determine the scaling for the parameters. Default is False.

        Returns:
            tuple: A tuple containing the initial parameters and the scaled field_true tensor.
        """
        field_init = field_true.clone()

        logger.debug(f"Initializing parameters with method: {method} {m} {field_scaling} {determine_scaling} {update_m_scaling} {rcond}")

        if self.scaling_init is not None and len(self.scaling_init) > 0:
            self.scaling = {k: v.clone() for k, v in self.scaling_init.items()}

        if field_scaling is not None:
            #self.field_scaling = field_scaling / torch.linalg.norm(field_true[:, self.axis_mask == 1])
            self.field_scaling = field_scaling / torch.linalg.vector_norm(field_init[:, self.axis_mask == 1], ord=2, dim=-1).mean()
            field_init = self.field_scaling * field_init

            logger.debug(f"Field scaling: {self.field_scaling}")

            if update_m_scaling and self.scaling["m"] is not None:
                self.scaling["m"] *= self.field_scaling
                logger.debug(f"Updated m scaling: {self.scaling['m'].tolist()}")

        if method == "pseudo_inv":
            # initialize m using inverse pass
            F = self.forward_model.forward_linear(r)
            if self.whitening_matrix is not None:
                F = torch.matmul(self.whitening_matrix, F)
            m = torch.linalg.lstsq(F, field_init[:, self.axis_mask == 1]).solution.view(field_init.shape[0], -1, 3)
        elif method == "pseudo_inv_regularized":
            # initialize m using inverse pass
            F = self.forward_model.forward_linear(r)
            if self.whitening_matrix is not None:
                F = torch.matmul(self.whitening_matrix, F)
            F_inv = torch.linalg.pinv(F, rcond=rcond)
            m = (F_inv @ field_init[:, self.axis_mask == 1].view(field_init.shape[0], F_inv.shape[-1],1)).view(field_init.shape[0], -1, 3)
        elif method == "random":
            # initialize m using random values
            m = torch.randn(field_init.shape[0], self.num_dipoles, 3, device=field_init.device)*1e-6
        elif method == "given":
            # initialize m using given parameters
            if m is None:
                raise ValueError("m must be provided when method is 'given'")
        else:
            raise NotImplementedError
        
        if determine_scaling:
            F = self.forward_model.forward_linear(r)
            if self.whitening_matrix is not None:
                F = torch.matmul(self.whitening_matrix, F)
            m_ = torch.linalg.lstsq(F, field_init[:, self.axis_mask == 1]).solution.view(field_init.shape[0], -1, 3)
            if "m" in self.scaling:
                self.scaling["m"] = self.scaling["m"] * torch.linalg.vector_norm(m_, ord=2, dim=(0,2), keepdim=True) #* self.field_scaling
                # fill zeros with mean
                non_zero_mean = torch.mean(self.scaling["m"]) if torch.count_nonzero(self.scaling["m"]) > 0 else 1
                self.scaling["m"] = torch.where(torch.isclose(self.scaling["m"], torch.zeros_like(self.scaling["m"]), atol=1e-9), non_zero_mean, self.scaling["m"])
    
                logger.debug(f"Determined m scaling: {self.scaling['m']}")

            if "r" in self.scaling:
                self.scaling["r"] = self.scaling["r"] * torch.linalg.vector_norm(torch.tensor(r, dtype=F.dtype, device=self.device), ord=2, dim=(0), keepdim=True)
                non_zero_mean = torch.mean(self.scaling["r"]) if torch.count_nonzero(self.scaling["r"]) > 0 else 1
                self.scaling["r"] = torch.where(torch.isclose(self.scaling["r"], torch.zeros_like(self.scaling["r"]), atol=1e-9), non_zero_mean, self.scaling["r"])
                logger.debug(f"Determined r scaling: {self.scaling['r']}")

        initial_parameters = self.parameters2parametervector(m, r)
        
       
        # Ensure parameters are leaf tensors that require gradients
        if isinstance(initial_parameters, list):
            initial_parameters = self._create_parameter_groups(initial_parameters)
        else:
            initial_parameters = torch.nn.Parameter(initial_parameters.detach().clone().requires_grad_(True))

        return initial_parameters, field_init

    def loss(self, parameters, field_true, warmup_done=False, return_numpy=False, **kwargs):

        # mask nan / non-recorded axes
        if field_true.ndim == 3:
            field_true = field_true[:, self.axis_mask == 1]
        if torch.isnan(field_true).any():
            raise ValueError("Field contains NaN values. Check if all non-recorded axes are masked.")

        # inverse pass
        m_hat, r_hat = self.parametervector2parameters(parameters)

        F = self.forward_model.forward_linear(r_hat)
        if self.whitening_matrix is not None:
            F = torch.matmul(self.whitening_matrix, F)
        T, S_D = field_true.shape

        # # Least squares solution
        if self.method == "twostep":
            #m_ = torch.linalg.lstsq(F, field_true.view(T, S_D, 1)).solution
            #m_hat = m_.view(T, -1, 3)
            F_inv = torch.linalg.pinv(F, rcond=1e-4)
            m_hat = (F_inv @ field_true.view(T,S_D,1)).view(T, -1, 3)

        # forward pass
        field_pred = torch.matmul(F, m_hat.view(T, -1, 1)).view(T, S_D)
        if field_pred.ndim == 3:
            field_pred = field_pred[:, self.axis_mask == 1]

        # compute loss & gradient
        loss_dict = {}
        #loss_dict["lncosh"] = lncosh(field_pred, field_true)
        #loss_dict["lhsaf"] = lhsaf(field_pred, field_true)
        loss_dict["mse"] = mse(field_pred, field_true, p=2)
        #loss_dict["mse"] = huber(field_pred, field_true)

        # rescale m and r for loss computation
        if "m" in self.scaling:
            m_hat = m_hat / self.scaling["m"]
        if "r" in self.scaling:
            r_hat = r_hat / self.scaling["r"]
        # if "lp_r" in self.loss_params:
        #     loss_dict["lp_r"] = norm(r_hat, **self.loss_params["lp_r"])
        # if "lp_m" in self.loss_params:
        #     loss_dict["lp_m"] = norm(m_hat, **self.loss_params["lp_m"])
        if "lp_r" in self.loss_params:
            loss_dict["lp_r_f"] = norm(r_hat[:,0], weight=self.loss_params["lp_r"]['weight'][0])
            loss_dict["lp_r_m"] = norm(r_hat[:,1], weight=self.loss_params["lp_r"]['weight'][1])
        if "lp_m" in self.loss_params:
            #m_hat_ = parameters[1].view(-1, self.num_dipoles, self.dof_per_dipole)
            loss_dict["lp_m_f"] = norm(m_hat[:,0], weight=self.loss_params["lp_m"]['weight'][0])
            loss_dict["lp_m_m"] = norm(m_hat[:,1], weight=self.loss_params["lp_m"]['weight'][1])


        # if "TV_r" in self.loss_params:
        #     loss_dict["TV_r"] = tv(r_hat, **self.loss_params["TV_r"])
        # if "TV_m" in self.loss_params:
        #     loss_dict["TV_m"] = tv(m_hat, **self.loss_params["TV_m"])
        if "TV_r" in self.loss_params:
            loss_dict["TV_r_f"] = tv(r_hat[:,0], weight=self.loss_params["TV_r"]['weight'][0])
            loss_dict["TV_r_m"] = tv(r_hat[:,1], weight=self.loss_params["TV_r"]['weight'][1])
        if "TV_m" in self.loss_params:
            loss_dict["TV_m_f"] = tv(m_hat[:,0], weight=self.loss_params["TV_m"]['weight'][0])
            loss_dict["TV_m_m"] = tv(m_hat[:,1], weight=self.loss_params["TV_m"]['weight'][1])



        if "BTV_r" in self.loss_params:
            loss_dict["BTV_r"] = bilateral_tv(r_hat, **self.loss_params["BTV_r"])
        if "BTV_m" in self.loss_params:
            loss_dict["BTV_m"] = bilateral_tv(m_hat, **self.loss_params["BTV_m"])
        if "corr_m" in self.loss_params:
            loss_dict["corr_m"] = correlation_penalty(m_hat, **self.loss_params["corr_m"]) if warmup_done else 0#/ loss_dict["mse"].detach()
        if "corr_r" in self.loss_params:
            loss_dict["corr_r"] = correlation_penalty(r_hat, **self.loss_params["corr_r"]) if warmup_done else 0

        # sum individual losses
        loss = sum(loss_dict.values())

        loss.backward()
        

        # log
        #S_ = torch.linalg.svdvals(F)
        self.callback.log(
            # parameters=(
            #     parameters.tolist() if isinstance(parameters, torch.Tensor) else [p.tolist() for p in parameters]
            # ),
            **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()},
            loss=loss.item(),
            #condition_number=(S_.max() / S_.min()).tolist(),
            #grad_m=parameters[0].grad.tolist(),
            #grad_r=parameters[1].grad.tolist(),
            **kwargs,
        )
        if return_numpy:
            return (loss.detach().cpu().numpy(), parameters.grad.detach().cpu().numpy())
        else:
            return loss

    def solve(
        self,
        field_true,
        initial_parameters,
        lr=dict(niter=100, start=1e-2, end=1e-2, warmup=0),
        loss_params=None,
        plot=True,
        savename=None,
        plt_param=False,
        early_stop=True,
        **kwargs,
    ):
        """
        Solves the inverse problem.
        Args:
            field_true (torch.Tensor): The measured field values.
            initial_parameters (torch.Tensor or list): Initial parameters for optimization.
            lr (dict): Learning rate schedule parameters with keys:
                - niter (int): Number of iterations.
                - start (float): Starting learning rate.
                - end (float): Ending learning rate.
                - warmup (int): Number of warmup iterations.
            loss_params (dict, optional): Parameters for the loss function. Defaults to None.
            plot (bool, optional): Whether to plot the results. Defaults to True.
            savename (str, optional): Filename to save the plot. Defaults to None.
            plt_param (bool, optional): Whether to plot parameters. Defaults to False.
            **kwargs: Additional arguments.

        Returns:
            tuple: Estimated parameters (m_hat, r_hat) as numpy arrays.
        """
        self.loss_params = loss_params if loss_params is not None else {}

        if not isinstance(field_true, torch.Tensor):
            field_true = torch.tensor(field_true, dtype=torch.float32, device=self.device)

        # initialize logging
        self.callback.reset()

        # initialize parameters
        if self.parameter_format == "vector":
            if not isinstance(initial_parameters, torch.Tensor):
                initial_parameters = torch.tensor(
                    initial_parameters, dtype=torch.float32, requires_grad=True, device=self.device
                )
            if initial_parameters.requires_grad is False:
                initial_parameters.requires_grad = True
            parameters = initial_parameters
        else:
            for i in range(len(initial_parameters)):
                if not isinstance(initial_parameters[i], torch.Tensor):
                    initial_parameters[i] = torch.tensor(
                        initial_parameters[i], dtype=torch.float32, requires_grad=True, device=self.device
                    )
                if initial_parameters[i].requires_grad is False:
                    initial_parameters[i].requires_grad = True
        parameters = initial_parameters

        if self.method in ["adam", "twostep", "lamb"]:
            # Setup optimizer parameter groups
            param_groups = self._setup_optimizer_groups(parameters, lr)

            optimizer = optim2.Lamb(param_groups, weight_decay=0.0) if self.method in ["lamb", "twostep"] else \
                optim.Adam(param_groups, weight_decay=0.0)
            
            # Setup individual schedulers for each parameter group
            schedulers = self._setup_schedulers(optimizer, lr)
            
            # Track best values 
            best_loss = float("inf")
            best_parameters = None
            best_iteration = 0
            
            # Early stopping variables (single counter for overall loss)
            early_stopping_counter = 0
            early_stopping_patience = lr.get("early_stopping_patience", 20) 
            early_stopping_threshold = lr.get("early_stopping_threshold", 1e-8)
            min_iter = lr.get("min_iter", 0)  # Minimum iterations before early stopping can trigger
            early_stopping_best_loss = float("inf")  # Separate tracking for early stopping
            
            # Cooldown variables - prevent counter updates after LR reduction
            cooldown_iterations = lr.get("cooldown_iterations", 10)  # Number of iterations to wait after LR reduction
            cooldown_remaining = 0  # Counter for remaining cooldown iterations

            with logging_redirect_tqdm():
                for i in (
                    pbar := trange(
                        lr["niter"] + lr["warmup"], desc="Iterations", leave=True, total=lr["niter"] + lr["warmup"]
                    )
                ):
                    prev_lr = optimizer.param_groups[0]["lr"]
                    
                    # Capture variables for closure nonlocal access
                    best_loss_captured = best_loss
                    best_parameters_captured = best_parameters
                    best_iteration_captured = best_iteration
                    def closure():
                        optimizer.zero_grad()
                        # Get learning rates using helper method
                        lr_dict = self._get_learning_rates_dict(optimizer)

                        #self.loss_params["corr_m"]["weight"] = penalty_scheduler(i)
                        #lr_dict["corr_m_weight"] = self.loss_params["corr_m"]["weight"]

                        loss = self.loss(parameters, field_true, warmup_done=(i>=schedulers["warmup_steps"]), **lr_dict)
                        loss_value = loss.tolist()

                        # if i==schedulers["warmup_steps"]-1:
                        #     self.loss_params["corr_m"]["weight"] *= loss_value

                        # Track best overall parameters
                        nonlocal best_loss_captured, best_parameters_captured, best_iteration_captured
                        if loss < best_loss_captured and i >= schedulers["warmup_steps"]:
                            best_loss_captured = loss
                            best_parameters_captured = [p.detach().clone() for p in parameters] if isinstance(parameters, list) else parameters.detach().clone()
                            best_iteration_captured = i
                            pbar.set_postfix({"Best Loss": loss_value})
                        
                        self.loss_value = loss_value  # Store for logging
                        return loss
                        
                    optimizer.step(closure=closure)
                    
                    # Update best parameters from closure
                    if best_loss_captured <  best_loss and i >= schedulers["warmup_steps"]:
                        best_loss = best_loss_captured
                        best_parameters = best_parameters_captured
                        best_iteration = best_iteration_captured
                    
                    # Step the appropriate schedulers
                    if i < schedulers["warmup_steps"]:
                        # During warmup, step all warmup schedulers
                        for scheduler in schedulers["warmup"]:
                            scheduler.step()
                    else:
                        # Store current LR to detect changes
                        prev_lrs = [group["lr"] for group in optimizer.param_groups]
                        
                        # After warmup, step all plateau schedulers with the same loss
                        for scheduler in schedulers["plateau"]:
                            scheduler.step(self.loss_value)
                        
                        # Check if any learning rate was reduced
                        current_lrs = [group["lr"] for group in optimizer.param_groups]
                        lr_reduced = any(current_lr < prev_lr for current_lr, prev_lr in zip(current_lrs, prev_lrs))
                        
                        if lr_reduced and cooldown_iterations > 0:
                            # LR was reduced, start cooldown period
                            cooldown_remaining = cooldown_iterations
                            logger.debug(f"LR reduced at iteration {i}, starting cooldown for {cooldown_iterations} iterations")
                        
                    # Simple early stopping logic based on overall loss
                    if early_stop and i > schedulers["warmup_steps"]:
                        # Update cooldown counter
                        if cooldown_remaining > 0:
                            cooldown_remaining -= 1
                        
                        if self.loss_value < early_stopping_best_loss - early_stopping_threshold:
                            # Loss improved significantly, reset counter and update early stopping best loss
                            early_stopping_best_loss = self.loss_value
                            early_stopping_counter = 0
                        elif cooldown_remaining > 0:
                            # During cooldown period, don't update counter
                            pass
                        else:
                            # No improvement and not in cooldown, increment counter
                            early_stopping_counter += 1
                        
                        # Stop if no improvement for patience iterations
                        if early_stop and early_stopping_counter >= early_stopping_patience and i >= min_iter:
                            logger.info(f"Early stopping at iteration {i} with best loss {best_loss}")
                            logger.info(f"Early stopping counter: {early_stopping_counter}")
                            break
                    else:
                        # During warmup, reset early stopping counter and cooldown
                        early_stopping_counter = 0
                        cooldown_remaining = 0
            # Store the best parameters for later use
            self.best_loss = best_loss
            self.best_parameters = best_parameters
            self.best_i = best_iteration
            
            logger.info(f"Best loss: {self.best_loss} at iteration {self.best_i}")


        elif self.method == "nested":
            print("nested")
            optim_r = optim.Adam(
                [{"params": parameters[1], "lr": 0.1 * lr["start"]}],
                weight_decay=0.0,
            )
            scheduler_r = optim.lr_scheduler.SequentialLR(
                optim_r,
                schedulers=[
                    # warmup cosine annealing
                    optim.lr_scheduler.LambdaLR(
                        optim_r, lr_lambda=lambda epoch: (1 - np.cos(np.pi * (epoch + 1) / max(lr["warmup"], 1))) / 2
                    ),
                    # optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch + 1) / lr["warmup"]),
                    optim.lr_scheduler.CosineAnnealingLR(optim_r, T_max=lr["niter"], eta_min=lr["end"]),
                ],
                milestones=[lr["warmup"]],
            )
            # optimize r
            self.best_parameters = None
            self.best_loss = float("inf")
            self.best_i = 0
            with logging_redirect_tqdm():
                for i in (
                    pbar := trange(
                        lr["niter"] + lr["warmup"], desc="Iterations", leave=True, total=lr["niter"] + lr["warmup"]
                    )
                ):
                    def closure_r():
                        # reset m optimizer and sheduler
                        optim_m = optim.Adam(
                            [{"params": parameters[0], "lr": lr["start"]}],
                            weight_decay=0.0,
                        )
                        scheduler_m = optim.lr_scheduler.SequentialLR(
                            optim_m,
                            schedulers=[
                                # warmup cosine annealing
                                optim.lr_scheduler.LambdaLR(
                                    optim_m, lr_lambda=lambda epoch: (1 - np.cos(np.pi * (epoch + 1) / max(lr["warmup"], 1))) / 2
                                ),
                                # optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch + 1) / lr["warmup"]),
                                optim.lr_scheduler.CosineAnnealingLR(optim_m, T_max=lr["niter"], eta_min=lr["end"]),
                            ],
                            milestones=[lr["warmup"]],
                        )

                        # optimize m
                        for j in range(100):
                            def closure_m():
                                optim_m.zero_grad()
                                loss = self.loss(parameters, field_true, lr=optim_m.param_groups[0]["lr"])
                                return loss
                            optim_m.step(closure_m)
                            scheduler_m.step()

                        # optimize r
                        optim_r.zero_grad()
                        loss = self.loss(parameters, field_true, lr=optim_r.param_groups[0]["lr"])
                        if loss < self.best_loss:
                            self.best_loss = loss
                            self.best_parameters = parameters
                            self.best_i = i
                            pbar.set_postfix({"Best Loss": loss.tolist()})
                        return loss
                    optim_r.step(closure_r)
                    scheduler_r.step()
            logger.info(f"Best loss: {self.best_loss} at iteration {self.best_i}")

        elif self.method == "basinhopping":
            if isinstance(parameters, torch.Tensor):
                parameters = parameters.detach().cpu().numpy()
            result = basinhopping(
                lambda x: self.loss(torch.tensor(x, dtype=torch.float32, device=self.device), field_true, return_numpy=True),
                parameters,
                minimizer_kwargs={
                    "jac": True,
                    "bounds": kwargs.pop("bnds", None),
                    "options": {"maxiter": 100},
                },  # , "options":{"ftol": 1e-6}
                # maxiter=maxiter,
                disp=True,
                # niter_success=2,
                interval=2,
                T=0,
                **kwargs,
            )
            self.best_parameters = torch.tensor(result.x, dtype=torch.float32, device=self.device)

        if plot:
            self.callback.plot(savename=savename, plt_param=plt_param)
        if self.best_parameters is not None:
            m_hat, r_hat = self.parametervector2parameters(self.best_parameters)
            if self.method == "twostep":
                F = self.forward_model.forward_linear(r_hat)
                if self.whitening_matrix is not None:
                    F = torch.matmul(self.whitening_matrix, F)
                m_hat = torch.linalg.lstsq(F, field_true[:, self.axis_mask==1].view(r_hat.shape[0], -1, 1)).solution.view(r_hat.shape[0], -1, 3)
        else:
            raise ValueError("Best parameters are not set. Ensure the optimization process has been completed successfully.")
        if self.field_scaling is not None:
            m_hat /= self.field_scaling
        return m_hat.detach().cpu().numpy(), r_hat.detach().cpu().numpy()

    def solve_with_config(self, field_true, initial_parameters, config, **kwargs):
        # Handle learning rates - use lr_groups if provided, otherwise use defaults
        lr_groups = config.get("lr_groups", [])
        
        # Handle patience groups - individual patience for each parameter group
        patience_groups = config.get("patience_groups", [])
        
        lr_args = dict(
            niter=config.get("niter", 200),
            start=config.get("lr", 1e-2),
            end=config.get("lr", 1e-2) * config.get("lr_end", 1e-2),
            warmup=config.get("warmup", 50),
            lr_groups=lr_groups,
            patience_groups=patience_groups,
            early_stopping_patience=config.get("early_stopping_patience", 20),
            early_stopping_threshold=config.get("early_stopping_threshold", 1e-8),
            min_iter=config.get("min_iter", 500),
            cooldown_iterations=config.get("cooldown_iterations", 10),
            pos_start=config.get("lr_pos", config.get("pos_start", config.get("lr", 1e-2)))
        )
        loss_args = {}
        if any(k in config for k in ["lp_m_weight_0", "lp_m_weight_1", "lp_m_p"]):
            loss_args["lp_m"] = dict(
            weight=[config.get("lp_m_weight_0", 0), config.get("lp_m_weight_1", 0)],
            p=config.get("lp_m_p", 2)
            )
        if any(k in config for k in ["lp_r_weight_0", "lp_r_weight_1", "lp_r_p"]):
            loss_args["lp_r"] = dict(
            weight=[config.get("lp_r_weight_0", 0), config.get("lp_r_weight_1", 0)],
            p=config.get("lp_r_p", 2)
            )
        if any(k in config for k in ["TV_m_weight_0", "TV_m_weight_1", "TV_m_p", "TV_m_n"]):
            loss_args["TV_m"] = dict(
            weight=[config.get("TV_m_weight_0", 0), config.get("TV_m_weight_1", 0)],
            p=config.get("TV_m_p", 2),
            n=config.get("TV_m_n", 1)
            )
        if any(k in config for k in ["TV_r_weight_0", "TV_r_weight_1", "TV_r_p", "TV_r_n"]):
            loss_args["TV_r"] = dict(
            weight=[config.get("TV_r_weight_0", 0), config.get("TV_r_weight_1", 0)],
            p=config.get("TV_r_p", 2),
            n=config.get("TV_r_n", 1)
            )
        if any(k in config for k in ["BTV_m_weight_0", "BTV_m_weight_1", "BTV_m_p", "BTV_m_sigma_d", "BTV_m_sigma_r"]):
            loss_args["BTV_m"] = dict(
            weight=[config.get("BTV_m_weight_0", 0), config.get("BTV_m_weight_1", 0)],
            p=config.get("BTV_m_p", 2),
            sigma_d=config.get("BTV_m_sigma_d", 1),
            sigma_r=config.get("BTV_m_sigma_r", 1)
            )
        if any(k in config for k in ["BTV_r_weight_0", "BTV_r_weight_1", "BTV_r_p", "BTV_r_sigma_d", "BTV_r_sigma_r"]):
            loss_args["BTV_r"] = dict(
            weight=[config.get("BTV_r_weight_0", 0), config.get("BTV_r_weight_1", 0)],
            p=config.get("BTV_r_p", 2),
            sigma_d=config.get("BTV_r_sigma_d", 1),
            sigma_r=config.get("BTV_r_sigma_r", 1)
            )
        if any(k in config for k in ["corr_m_weight"]):
            loss_args["corr_m"] = dict(
            weight=config.get("corr_m_weight", 0),
            )
        if any(k in config for k in ["corr_r_weight"]):
            loss_args["corr_r"] = dict(
            weight=config.get("corr_r_weight", 0),
            )
        m_hat, r_hat = self.solve(
            field_true,
            #method="adam",
            initial_parameters=initial_parameters,
            lr=lr_args,
            loss_params=loss_args,
            **kwargs,
        )
        return m_hat, r_hat

    def calculate_and_plot_errors(
        self,
        m_true,
        r_true,
        m_pred,
        r_pred,
        time,
        field_true=None,
        field_pred=None,
        print_params=False,
        plot=True,
        savename=None,
        alpha=0.7,
        labels=["Fetal", "Maternal"]
    ):
    # Convert the parameters to numpy arrays.
        if m_true is not None:
            if isinstance(m_true, torch.Tensor):
                m_true = m_true.detach().cpu().numpy()
        if r_true is not None:
            if isinstance(r_true, torch.Tensor):
                r_true = r_true.detach().cpu().numpy()
        if isinstance(m_pred, torch.Tensor):
            m_pred = m_pred.detach().cpu().numpy()
        if isinstance(r_pred, torch.Tensor):
            r_pred = r_pred.detach().cpu().numpy()
        if field_true is not None and isinstance(field_true, torch.Tensor):
            field_true = field_true.detach().cpu().numpy()
        if field_pred is not None and isinstance(field_pred, torch.Tensor):
            field_pred = field_pred.detach().cpu().numpy()

        # If true parameters are provided, perform full error computations.
        if m_true is not None and r_true is not None:
            if r_true.ndim == 2:
                r_true = r_true[np.newaxis, :, :]
            if r_true.shape[0] == 1:
                r_true = np.repeat(r_true, len(time), axis=0)
            if m_true.ndim == 2:
                m_true = m_true[np.newaxis, :, :]
            if m_true.shape[0] == 1:
                m_true = np.repeat(m_true, len(time), axis=0)

            def order_params(p_true, p_pred):
                """Order the predicted parameters according to the true parameters."""
                cost_matrix = np.linalg.norm(p_true[0, :, np.newaxis] - p_pred[0], axis=-1)
                row_ind_m, col_ind_m = linear_sum_assignment(cost_matrix)
                pred_ordered = p_pred[:, col_ind_m]
                return pred_ordered

            # Order dipoles.
            m_pred = order_params(m_true, m_pred)
            r_pred = order_params(r_true, r_pred)

            if print_params:
                for i in range(self.num_dipoles):
                    print(f"Pred. Dipole {i+1}: \t m = {m_pred[:, i]} \t r = {r_pred[:, i]}")
                    print(f"True Dipole {i+1}: \t m = {m_true[:, i]} \t r = {r_true[:, i]}")

            # Calculate the relative errors
            if field_true is None:
                field_true = self.forward_model.forward_linear(r_true, m_true, as_numpy=True)
            if field_pred is None:
                field_pred = self.forward_model.forward_linear(r_pred, m_pred, as_numpy=True)
            absolute_field_error = np.abs((field_true - field_pred))
            absolute_m_error = np.abs((m_true - m_pred))
            absolute_r_error = np.abs((r_true - r_pred))

            if plot:
                plt.figure()
                plt.plot(
                    time,
                    absolute_field_error if absolute_field_error.ndim == 1 else absolute_field_error.mean(axis=(-2, -1)),
                    label=r"Field $B$",
                )
                plt.plot(time, absolute_m_error.mean(axis=(-2, -1)), label=r"Dipole Moment $m$")
                plt.plot(time, absolute_r_error.mean(axis=(-2, -1)), label=r"Position $r$")
                plt.xlabel("Time [s]")
                plt.ylabel("Mean Absolute Error")
                plt.yscale("log")
                plt.grid()
                plt.legend(loc="upper right", fancybox=False)
                if savename:
                    plt.savefig(f"{savename.split('.pdf')[0]}_err.pdf", dpi=500, bbox_inches="tight", pad_inches=0)
                    plt.close()
                plt.show()

            print("\n#####################################################################")
            print(f"Absolute Field Error: {absolute_field_error.mean()}")

        if plot:
            fig, axs = plt.subplots(2, self.num_dipoles, figsize=(7.11, 3.5), dpi=500, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
            if axs.ndim == 1:
                axs = axs[:, np.newaxis]
            axis_label = [r"$x$", r"$y$", r"$z$"]
        for k in range(self.num_dipoles):
            if m_true is not None and r_true is not None:
                print(f"Dipole {k+1}: Absolute Error M: {absolute_m_error[:, k].mean()}")
                print(f"Dipole {k+1}: Absolute Error r: {absolute_r_error[:, k].mean()}")

            if plot:
                # Plot magnetic moments
                for i in range(m_pred.shape[-1]):
                    if m_true is not None:
                        axs[0, k].plot(time, m_true[:, k, i].flatten(), "--", label=f"True {axis_label[i]}", color=f"C{i}")
                    axs[0, k].plot(
                        time, m_pred[:, k, i].flatten(), label=f"Pred. {axis_label[i]}", color=f"C{i}", alpha=alpha
                    )
                axs[0, k].grid(True)
                axs[0, k].set_title(f"{labels[k]}")

                # Plot positions
                for i in range(m_pred.shape[-1]):
                    if r_true is not None:
                        axs[1, k].plot(time, r_true[:, k, i].flatten(), "--", label=f"True {axis_label[i]}", color=f"C{i}")
                    axs[1, k].plot(
                        time, r_pred[:, k, i].flatten(), label=f"Pred. {axis_label[i]}", color=f"C{i}", alpha=alpha
                    )
                axs[1, k].set_xlabel("Time [s]")
                axs[1, k].grid(True)

                axs[0, 0].set_ylabel(r"Magnetic Moment [$\mathrm{\mu}$Am$^2$]")
                axs[1, 0].set_ylabel("Position [m]")
                if r_true is not None:
                    all_vals = np.concatenate((r_true[:, k, :], r_pred[:, k, :]))
                else:
                    all_vals = r_pred[:, k, :].copy()
                min_val = np.min(all_vals)
                max_val = np.max(all_vals)
                axs[1, k].set_ylim(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val))

                handles, labels_ = axs[0, 0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels_,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.07),
                    ncol=3 * self.num_dipoles,
                    fancybox=False,
                )

                plt.tight_layout()

        if plot and savename:
            plt.savefig(f"{savename.split('.pdf')[0]}_fit_all_dipoles.pdf", dpi=500, bbox_inches="tight", pad_inches=.01)
            plt.close()

        if m_true is not None and r_true is not None:
            print(f"Total Dipole Moment Error: {absolute_m_error.mean()}")
            print(f"Total Position Error: {absolute_r_error.mean()}")
            print(f"Total Dipole Moment Error: {absolute_m_error.mean(axis=(0,-1))}")
            print(f"Total Position Error: {absolute_r_error.mean(axis=(0,-1))}")
            return (
                absolute_field_error.mean(),
                absolute_m_error.mean(axis=(0, -1)).mean(),
                absolute_r_error.mean(axis=(0, -1)).mean(),
            )