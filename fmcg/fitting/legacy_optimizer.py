from scipy.optimize import basinhopping, linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch_optimizer as optim2
from tqdm import trange

from utils.utils import Logger


class Optimizer:
    def __init__(self, model, field_true, num_dipoles=1, num_time=1):
        self.model = model
        self.field_true = field_true
        self.num_dipoles = num_dipoles
        self.num_time = num_time

        self.parameter_labels = [
            f"{label}_{i+1}"
            for i in range(num_dipoles)
            for label in [r"$m_x$", r"$m_y$", r"$m_z$", r"$r_x$", r"$r_y$", r"$r_z$"]
        ]
        self.num_params = model.dof_per_dipole * self.num_dipoles * self.num_time

        self.callback = Logger()

    def regularization(self, parameters):
        m, r = self.model.parametervector2parameters(parameters, self.num_dipoles)
        return np.sum(np.linalg.norm(np.diff(m, axis=0), axis=0) ** 2) + np.sum(
            np.linalg.norm(np.diff(r, axis=0) ** 2, axis=0) ** 2
        )

    def cost_function(self, parameters):
        c = np.sum(
            np.linalg.norm(
                self.model.optimizer_forward_pass(parameters, self.num_dipoles) - self.field_true, axis=(1, 2)
            )
            ** 2
        ) + self.regularization(parameters)
        # self.callback.log_3(parameters, c, accept=False)
        return c

    def solve(self, initial_parameters=None, bnds=None, maxiter=1000, plot=True, **kwargs):
        if initial_parameters is None:
            # initial_parameters = np.ones(self.num_params) * 0.04
            initial_parameters = np.random.rand(self.num_params) / 10
            # initial_parameters = [-5.001e-03,-1.000e-02,9.972e-05,-4.000e-02,4.000e-02,1.400e-01,-4.999e-03,5.000e-03,9.990e-04,4.000e-02,1.400e-01,3.999e-02]
            print(f"Initial parameters: {initial_parameters}")

        result = basinhopping(
            self.cost_function,
            initial_parameters,
            # minimizer_kwargs={"method": "Nelder-Mead", "options": {"maxiter": 5000}},
            # bounds=bnds,
            # maxiter=maxiter,
            callback=self.callback.log_3,
            disp=True,
            # stepsize=.01,
            # niter_success=2,
            **kwargs,
        )

        # result = minimize(
        #     self.cost_function,
        #     initial_parameters,
        #     bounds=bnds,
        #     method="Nelder-Mead",
        #     callback=self.callback.log,
        #     options={"maxiter": maxiter},
        #     **kwargs,
        # )
        if plot:
            self.callback.plot(self.parameter_labels)

            m_dipoles, r_dipoles = self.model.parametervector2parameters(result.x, self.num_dipoles)
            for i in range(self.num_dipoles):
                print(f"Dipole {i+1}:")
                print(f"m = {m_dipoles[:,i]} \t r = {r_dipoles[:,i]}")

        return result


class OptimizerTorch(Optimizer):
    def __init__(
        self,
        model,
        field_true,
        num_dipoles=1,
        num_time=1,
        grad_norm=False,
        data_loss_ord=2,
        n_diff_TV=1,
        reg_weight_variance=0,
        reg_weight=np.array([0.5, 0.5, 0.5, 1, 1, 1]) * 50,
        TV_ord=2,
    ):
        if not isinstance(field_true, torch.Tensor):
            field_true = torch.tensor(field_true, dtype=torch.float32, device=model.device)
        super().__init__(model, field_true, num_dipoles, num_time)
        self.callback = Logger(parameter_labels=self.parameter_labels)
        self.grad_norm = grad_norm
        self.reg_weight = reg_weight
        self.data_loss_ord = data_loss_ord
        self.n_diff_TV = n_diff_TV
        self.reg_variance = reg_weight_variance
        self.TV_ord = TV_ord

    def regularization(self, parameters, n=1):
        # m, r = self.model.parametervector2parameters(parameters, self.num_dipoles)
        # reg_m = torch.sum(torch.norm(torch.diff(m, dim=0), dim=0, p=2))
        # reg_r = torch.sum(torch.norm(torch.diff(r, dim=0), dim=0, p=2))
        # return reg_m + 10 * reg_r

        # reg = torch.linalg.vector_norm(torch.diff(parameters.view(-1, self.num_dipoles, self.model.dof_per_dipole), dim=0), dim=0).mean()

        # reg = torch.norm(torch.diff(parameters.view(-1, self.num_dipoles, self.model.dof_per_dipole), dim=0))
        weight = torch.Tensor(self.reg_weight).unsqueeze(0).unsqueeze(0).to(self.model.device)
        reg = (
            weight
            * (
                torch.linalg.vector_norm(
                    torch.diff(parameters.view(-1, self.num_dipoles, self.model.dof_per_dipole), dim=0, n=n),
                    dim=0,
                    ord=self.TV_ord,
                )
                # / torch.linalg.vector_norm(
                #     parameters.view(-1, self.num_dipoles, self.model.dof_per_dipole), dim=(0,2), ord=2, keepdim=True
                # )
            )
            # ** 2
        ).sum()

        return reg

    def mse(self, field_pred, field_true=None):
        if field_true is None:
            field_true = self.field_true
        field_pred = field_pred[:, self.model.axis_mask == 1]
        field_true = field_true[:, self.model.axis_mask == 1]
        return (
            torch.linalg.vector_norm(field_pred - field_true, ord=self.data_loss_ord) ** self.data_loss_ord
        )

    def cost_function(self, parameters, return_numpy=True, **kwargs):
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.tensor(parameters, dtype=torch.float32, requires_grad=True, device=self.model.device)

        # forward pass
        field_pred = self.model.optimizer_forward_pass(parameters, self.num_dipoles)

        # compute loss & gradient
        # c = self.loss(parameters, field_pred)
        mse = self.mse(field_pred)  # / self.num_time
        reg = (
            self.regularization(parameters, n=self.n_diff_TV)
            if self.num_time > 1
            else torch.tensor([0], device=self.model.device)
        )

        p = parameters.view(-1, self.num_dipoles, self.model.dof_per_dipole)
        reg2 = self.reg_variance * torch.linalg.vector_norm(p - p.mean(dim=0), ord=1, dim=0).sum()
        # reg2 = 1 * self.regularization(parameters, n=2)
        c = mse + reg + reg2
        c.backward()

        if self.grad_norm:
            self._grad_norm_paramgroups(parameters)
        # log
        self.callback.log(
            parameters=parameters.tolist(),
            mse=mse.tolist(),
            loss=c.tolist(),
            regularization=reg.tolist(),
            grad=parameters.grad.tolist(),
            reg2=reg2.tolist(),
            **kwargs,
        )
        if return_numpy:
            return c.data.cpu().numpy(), parameters.grad.data.cpu().numpy()
        else:
            return c

    def solve(
        self,
        stepsize=1,
        initial_parameters=None,
        bnds=None,
        maxiter=1000,
        plot=True,
        savename=None,
        reg_weight=None,
        **kwargs,
    ):

        self.callback.reset()
        if initial_parameters is None:
            # initial_parameters = np.ones(self.num_params) * 0.04
            initial_parameters = np.random.rand(self.num_params) / 10

        print(f"Initial parameters: {initial_parameters}")
        print(f"Bounds : {bnds}")

        if reg_weight is not None:
            self.reg_weight = reg_weight

        # bnds = Bounds(lb=bnds[:, 0], ub=bnds[:, 1]) if bnds is not None else None
        # result = dual_annealing(
        #     self.cost_function,
        #     bounds=bnds,
        #     minimizer_kwargs={"jac": False, "args": {"maxiter": 1000}},  # , "options":{"ftol": 1e-6}
        #     ## maxiter=maxiter,
        #     #disp=True,
        #     #stepsize=stepsize,
        #     ## niter_success=2,
        #     #interval=2,
        #     #T=0,
        #     **kwargs,
        # )
        # result = shgo(
        #     self.cost_function,
        #     bnds,
        #     minimizer_kwargs={"jac": True, "method":'l-bfgs-b', "bounds": bnds},
        #     options={"jac": True},
        #     sampling_method='sobol',
        #     iters=10,
        # )
        result = basinhopping(
            self.cost_function,
            initial_parameters,
            minimizer_kwargs={"jac": True, "bounds": bnds, "args": {"maxiter": 500}},  # , "options":{"ftol": 1e-6}
            # maxiter=maxiter,
            disp=True,
            stepsize=stepsize,
            # niter_success=2,
            interval=2,
            T=0,
            **kwargs,
        )

        if plot:
            self.callback.plot(savename=savename)
        return self.model.parametervector2parameters(result.x, self.num_dipoles)

    def _grad_norm_paramgroups(self, parameters):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        for p in parameters:
            if p.grad is None:
                continue
            p.grad /= torch.linalg.vector_norm(p.grad, ord=2)

            # p_grad = p.grad.view(-1, self.num_dipoles, self.model.dof_per_dipole)
            # p_grad /= torch.linalg.vector_norm(p_grad, ord=2, dim=(0), keepdims=True)

    def solve_adam(
        self,
        maxiter=100,
        lr=0.01,
        i_warmup=500,
        eta_min=0.000001,
        initial_parameters=None,
        plot=True,
        savename=None,
        plt_param=False,
        reg_weight=None,
        method="adam",
        optimizer_params_import=False,
    ):

        self.callback.reset()

        if initial_parameters is None:
            initial_parameters = np.random.rand(self.num_params) / 10
        # print(f"Initial parameters: {initial_parameters}")

        if reg_weight is not None:
            self.reg_weight = reg_weight

        parameters = torch.tensor(
            initial_parameters, dtype=torch.float32, requires_grad=True, device=self.model.device
        )
        if method == "adam":
            optimizer = optim.Adam([parameters], lr=lr, weight_decay=0.0)
        elif method == "yogi":
            optimizer = optim2.Yogi(
                [parameters],
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-3,
                initial_accumulator=1e-6,
                weight_decay=0,
            )
        elif method == "adamp":
            optimizer = optim2.AdamP(
                [parameters], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, delta=0.1, wd_ratio=0.1
            )
        # optimizer = optim.Adamax([parameters], lr=lr, weight_decay=0.0)
        # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=maxiter)
        # scheduler = optim.lr_scheduler.SequentialLR(
        #     optimizer,
        #     schedulers=[
        #     optim.lr_scheduler.LinearLR(optimizer, start_factor=.1, total_iters=i_warmup),
        #     optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maxiter-i_warmup, eta_min=eta_min),
        #     ],
        #     milestones=[i_warmup],
        # )
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2500, T_mult=1, eta_min=0.0001)
        self.optimizer = optimizer
        if optimizer_params_import:
            def load_state_dict(optimizer, state_dict):
                state_dict_ = optimizer.state_dict()
                state_dict_['state'] = state_dict['state']
                return state_dict_
            optimizer.register_load_state_dict_pre_hook(load_state_dict)
            optimizer.load_state_dict(optimizer_params_import)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maxiter, eta_min=eta_min)

        self.best_parameters = None
        self.best_loss = float("inf")
        self.best_i = 0
        for i in (pbar := trange(maxiter, desc="Iterations", leave=True, total=maxiter)):

            def closure():
                optimizer.zero_grad()
                loss = self.cost_function(parameters, return_numpy=False, lr=optimizer.param_groups[0]["lr"])
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_parameters = parameters
                    self.best_i = i
                    pbar.set_postfix({"Best Loss": loss.tolist()})
                return loss

            optimizer.step(closure)
            #parameters.grad = None
            scheduler.step()
        print(f"Best loss: {self.best_loss} at iteration {self.best_i}")

        if plot:
            self.callback.plot(savename=savename, plt_param=plt_param)
        return self.model.parametervector2parameters(self.best_parameters.detach().cpu().numpy(), self.num_dipoles)

    def calculate_and_plot_errors(
        self, m_true, r_true, m_pred, r_pred, time, print_params=False, plot=True, savename=None
    ):
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

        # order dipoles
        m_pred = order_params(m_true, m_pred)
        r_pred = order_params(r_true, r_pred)

        if print_params:
            for i in range(self.num_dipoles):
                print(f"Pred. Dipole {i+1}: \t m = {m_pred[:,i]} \t r = {r_pred[:,i]}")
                print(f"True Dipole {i+1}: \t m = {m_true[:,i]} \t r = {r_true[:,i]}")

        # Calculate the relative errors
        field_true = self.model.forward_temporal(m_true, r_true)
        field_pred = self.model.forward_temporal(m_pred, r_pred)
        relative_error_field = (
            torch.abs((field_true - field_pred) / field_true).detach().cpu().numpy() * 100
        )
        relative_error_m = np.abs((m_true - m_pred) / m_true) * 100
        relative_error_r = np.abs((r_true - r_pred) / r_true) * 100

        if plot:
            plt.figure()
            plt.plot(
                time,
                relative_error_field if relative_error_field.ndim == 1 else relative_error_field.mean(axis=(-2,-1)),
                label=r"Field $B$",
            )
            plt.plot(time, relative_error_m.mean(axis=(-2,-1)), label=r"Dipole Moment $m$")
            plt.plot(time, relative_error_r.mean(axis=(-2,-1)), label=r"Position $r$")
            plt.xlabel("Time [s]")
            plt.ylabel("Relative Error [%]")
            plt.yscale("log")
            plt.grid()
            plt.legend(loc="upper right", fancybox=False)
            if savename:
                plt.savefig(f"{savename.split('.pdf')[0]}_err.pdf", dpi=500, bbox_inches="tight", pad_inches=0)
            plt.show()

        print("\n#####################################################################")
        print(f"Relative Field Error: {relative_error_field.mean():.2f}%")

        for k in range(self.num_dipoles):
            print(f"Dipole {k+1}: Relative Error M: {relative_error_m[:, k].mean():.2f}%")
            print(f"Dipole {k+1}: Relative Error r: {relative_error_r[:, k].mean():.2f}%")

            if plot:
                # Plot parameters against time
                axis_label = [r"$_x$", r"$_y$", r"$_z$"]
                fig, axs = plt.subplots(1, 2, figsize=(7.11, 2.5), sharex=True, dpi=500)

                for i in range(m_true.shape[-1]):
                    axs[0].plot(
                        time, m_true[:, k, i].flatten(), linestyle="--", label=f"True m{axis_label[i]}", color=f"C{i}"
                    )
                    axs[0].plot(
                        time, m_pred[:, k, i].flatten(), label=f"Pred. m{axis_label[i]}", color=f"C{i}", alpha=0.7
                    )
                axs[0].set_xlabel("Time [s]")
                axs[0].set_ylabel(r"Magnetic Moment [$\mathrm{\mu}$Am$^2$]")
                axs[0].grid()
                axs[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.6), fancybox=False)

                for i in range(m_true.shape[-1]):
                    axs[1].plot(
                        time, r_true[:, k, i].flatten(), linestyle="--", label=f"True r{axis_label[i]}", color=f"C{i}"
                    )
                    axs[1].plot(
                        time, r_pred[:, k, i].flatten(), label=f"Pred. r{axis_label[i]}", color=f"C{i}", alpha=0.7
                    )
                axs[1].set_xlabel("Time [s]")
                axs[1].set_ylabel(f"Position [m]")
                axs[1].grid()
                axs[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.6), fancybox=False)

                fig.suptitle(f"Dipole {k+1}")
                plt.tight_layout()
                fig.subplots_adjust(top=0.65, right=1, wspace=0.25)
                if savename:
                    plt.savefig(f"{savename.split('.pdf')[0]}_fit{k}.pdf", dpi=750, bbox_inches="tight", pad_inches=0)
                plt.show()

        return relative_error_field.mean(), relative_error_m.mean(), relative_error_r.mean()
