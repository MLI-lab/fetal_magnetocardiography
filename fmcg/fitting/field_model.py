import matplotlib.pyplot as plt
import numpy as np
import torch

from ..utils.plotting import plot_utils
import logging
import warnings


class ForwardModel:
    """
    FieldModel implements a magnetic field model using magnetic dipoles and computes
    the resulting fields at given sensor locations using various methods (temporal, linear, numpy).

    Attributes:
        r_sensors (torch.Tensor or array-like): Coordinates of the sensors.
        device (str): The computing device (e.g., "cpu", "cuda").
        r_scaling (torch.Tensor): Scaling factor for dipole positions.
        m_scaling (torch.Tensor): Scaling factor for dipole moments.
        axis_mask (torch.Tensor): A mask applied to the sensor axes for plotting (expected shape: (16, 3)).

    """

    def __init__(
        self, r_sensors, r_scaling=1, m_scaling=1, device="cpu", axis_mask=None
    ):
        if not isinstance(r_sensors, torch.Tensor):
            r_sensors = torch.tensor(r_sensors, dtype=torch.float32, device=device)
        self.r_sensors = r_sensors
        self.dof_per_dipole = 6
        self.device = device
        if not isinstance(r_scaling, torch.Tensor):
            r_scaling = torch.tensor(r_scaling, dtype=torch.float32, device=device)
        if not isinstance(m_scaling, torch.Tensor):
            m_scaling = torch.tensor(m_scaling, dtype=torch.float32, device=device)
        if r_scaling.ndim == 1:
            r_scaling = r_scaling.unsqueeze(-1)
        if m_scaling.ndim == 1:
            m_scaling = m_scaling.unsqueeze(-1)

        self.r_scaling = r_scaling
        self.m_scaling = m_scaling

        if axis_mask is None:
            axis_mask = torch.ones((16, 3), dtype=torch.float32, device=device)
        if not isinstance(axis_mask, torch.Tensor):
            axis_mask = torch.tensor(axis_mask, dtype=torch.float32, device=device)
        # axis_mask[axis_mask == 0] = float('nan')
        self.axis_mask = axis_mask
        assert axis_mask.shape == (16, 3)

    def forward_linear(self, r_dipoles, m_dipoles=None, as_numpy=False, silent=False):
        """
        Compute the linear magnetic field operator determined by the displacement vectors r_dipoles. If corresponding
        magnetic moments m_dipoles are provided, the magnetic field is returned, obtained by multiplying the operator
        with the magnetic moments. The field is returned in uT.

        **Note:** This produces the same results as `forward_temporal` when `m_dipoles` and `r_dipoles` are provided.
        However, unlike `forward_temporal`, which computes the result in a compact form to potentially reduce memory usage,
        this implementation constructs the linear operator explicitly.


        Parameters
        -----------
        r_dipoles : array-like or torch.Tensor
            Positions of the dipoles. Should be of shape (D, 3) or (T, D, 3), where D is the number of dipoles
            and T is the number of time steps. If not a torch.Tensor, it will be converted to one.
        m_dipoles : array-like or torch.Tensor, optional
            Magnetic moments of the dipoles. Should be of shape (D, 3) or (T, D, 3). If not provided, the function
            will return the magnetic field operator matrix. If not a torch.Tensor, it will be converted to one.
        as_numpy : bool, optional
            If True, the output will be converted to a NumPy array. Default is False.

        Returns:
        --------
        torch.Tensor
            If `m_dipoles` is provided, returns the magnetic field tensor of shape (T, S, 3), where S is the number
            of sensors. If `m_dipoles` is not provided, returns the magnetic field operator matrix of shape (T, S*3, D*3).
        """
        if not isinstance(r_dipoles, torch.Tensor) and not silent:
            r_dipoles = torch.tensor(
                r_dipoles.copy(), dtype=torch.float32, device=self.device
            )
            logging.debug(
                "Warning: not a tensor. Continuing without tracking gradients."
            )

        assert r_dipoles.shape[-1] == 3

        if r_dipoles.ndim == 1:
            r_dipoles = r_dipoles.reshape((1, 3))
        if r_dipoles.ndim == 2:
            r_dipoles = r_dipoles.reshape((1, -1, 3))

        if m_dipoles is not None:
            if not isinstance(m_dipoles, torch.Tensor) and not silent:
                m_dipoles = torch.tensor(
                    m_dipoles.copy(), dtype=torch.float32, device=self.device
                )
                logging.debug(
                    "Warning: not a tensor. Continuing without tracking gradients."
                )

            assert m_dipoles.shape[-1] == 3

            if m_dipoles.ndim == 1:
                m_dipoles = m_dipoles.reshape((1, 3))
            if m_dipoles.ndim == 2:
                m_dipoles = m_dipoles.reshape((1, -1, 3))

            assert m_dipoles.ndim == 3 and r_dipoles.ndim == 3
            assert m_dipoles.shape[-2] == r_dipoles.shape[-2]

            T, D, _ = m_dipoles.shape  # [T x D x 3]
            m_vec = m_dipoles.reshape(T, D * 3, 1)  # [T x D*3 x 1]

        # Compute displacement vectors and norms
        r = self.r_sensors.reshape(1, -1, 1, 3) - r_dipoles.reshape(
            r_dipoles.shape[0], 1, -1, 3
        )  # [T x S x D x 3]
        r_norm = torch.norm(r, dim=-1, keepdim=True)  # [T x S x D x 1]
        r_hat = r / r_norm  # unit vector, same shape as r

        # Compute linear magnetic field operator
        mu0_4pi = 0.1  # mu0 / (4 * pi);  in uT | mu0 = 4pi * 10^-7 H/m
        r_outer = r_hat.unsqueeze(-1) * r_hat.unsqueeze(-2)  # [T x S x D x 3 x 3]
        I = torch.eye(3, device=r.device).view(1, 1, 1, 3, 3)  # [1 x 1 x 1 x 3 x 3]
        A = (
            mu0_4pi * (3 * r_outer - I) / (r_norm**3).unsqueeze(-1)
        )  # [T x S x D x 3 x 3]

        # Compute magnetic field linear by reshaping into vectorized representation
        T, S, D, _, _ = A.shape
        A_mat = A.permute(0, 1, 3, 2, 4).reshape(T, S * 3, D * 3)  # [T x S*3 x D*3]

        if m_dipoles is None:
            if self.axis_mask is not None:
                A_mat = A_mat[:, self.axis_mask.flatten() == 1]

            if as_numpy:
                A_mat = A_mat.detach().cpu().numpy()
            return A_mat

        B_vec = torch.matmul(A_mat, m_vec)  # [T x S*3 x 1]
        B_total = B_vec.reshape(T, S, 3)  # [T x S x 3]
        if as_numpy:
            B_total = B_total.detach().cpu().numpy()
        return B_total

    def forward_temporal(self, m_dipoles, r_dipoles):
        """
        Compute the magnetic field at sensor locations due to dipoles over time. The field is returned in uT.

        **Note:** This produces the same results as `forward_linear` when `m_dipoles` and `r_dipoles` are provided.
        However, unlike `forward_linear`, which constructs the linear operator explicitly, this implementation performs
        the computation in a more compact form, potentially reducing memory usage.

        Parameters:
        -----------
        m_dipoles : array-like or torch.Tensor
            Magnetic moments of the dipoles. Should have shape (n_time, n_dipoles, 3) or (n_dipoles, 3).
        r_dipoles : array-like or torch.Tensor
            Positions of the dipoles. Should have shape (n_time, n_dipoles, 3) or (n_dipoles, 3).

        Returns:
        --------
        torch.Tensor
            Magnetic field at the sensor locations. Shape is (n_time, n_sensors, 3).

        """
        if not isinstance(self.r_sensors, torch.Tensor):
            self.r_sensors = torch.tensor(
                self.r_sensors, dtype=torch.float32, device=self.device
            )
        if not isinstance(m_dipoles, torch.Tensor):
            m_dipoles = torch.tensor(
                m_dipoles.copy(), dtype=torch.float32, device=self.device
            )
            logging.debug(
                "Warning: not a tensor. Continuing without tracking gradients."
            )
        if not isinstance(r_dipoles, torch.Tensor):
            r_dipoles = torch.tensor(
                r_dipoles.copy(), dtype=torch.float32, device=self.device
            )
            logging.debug("Not a tensor. Continuing without tracking gradients.")

        assert m_dipoles.shape[-1] == 3 and r_dipoles.shape[-1] == 3

        if m_dipoles.ndim == 1:
            m_dipoles = m_dipoles.reshape((1, 3))
        if m_dipoles.ndim == 2:
            m_dipoles = m_dipoles.reshape((1, -1, 3))
        if r_dipoles.ndim == 1:
            r_dipoles = r_dipoles.reshape((1, 3))
        if r_dipoles.ndim == 2:
            r_dipoles = r_dipoles.reshape((1, -1, 3))

        assert m_dipoles.ndim == 3 and r_dipoles.ndim == 3
        assert m_dipoles.shape[-2] == r_dipoles.shape[-2]

        # Calculate the relative position vectors from each sensor to each dipole at each time point
        r = self.r_sensors.reshape(1, -1, 1, 3) - r_dipoles.reshape(
            r_dipoles.shape[0], 1, -1, 3
        )
        # Reshape m_dipoles for broadcasting
        m_dipoles = m_dipoles.reshape(m_dipoles.shape[0], 1, -1, 3)
        # Magnetic constant in uT
        mu0_4pi = 0.1  # in uT | mu0 = 4pi * 10^-7 H/m
        # Normalize the relative position vectors
        r_norm = torch.norm(
            r, dim=-1, keepdim=True
        )  # [n_time x n_sensors x n_dipoles x 1]
        r_hat = r / r_norm
        # Calculate the dot product
        dot = torch.sum(
            m_dipoles * r_hat, dim=-1, keepdim=True
        )  # [n_time x n_sensors x n_dipoles x 1]
        # Calculate the magnetic field using the formula for the magnetic field of a dipole
        B = mu0_4pi * (3 * r_hat * dot - m_dipoles) / r_norm**3
        # Sum the contributions from all dipoles
        return torch.sum(B, dim=-2)  # [n_time x n_sensors x 3]

    def forward_numpy(self, m_dipoles, r_dipoles):
        """
        Computes the forward temporal model and returns the result as a NumPy array.

        Args:
            m_dipoles (array-like or torch.Tensor): The magnetic dipole moments.
            r_dipoles (array-like or torch.Tensor): The positions of the dipoles.

        Returns:
            numpy.ndarray: The result of the forward temporal model computation.
        """
        if not isinstance(m_dipoles, torch.Tensor):
            m_dipoles = torch.tensor(m_dipoles, dtype=torch.float32, device=self.device)
        if not isinstance(r_dipoles, torch.Tensor):
            r_dipoles = torch.tensor(r_dipoles, dtype=torch.float32, device=self.device)
        return self.forward_temporal(m_dipoles, r_dipoles).detach().cpu().numpy()

    def to(self, device):
        """
        Moves the model's parameters to the specified device.

        Args:
            device (torch.device): The device to move the parameters to (e.g., 'cpu' or 'cuda').

        Returns:
            None
        """
        self.device = device
        self.r_sensors = self.r_sensors.to(device)
        self.m_scaling = self.m_scaling.to(device)
        self.r_scaling = self.r_scaling.to(device)
        # self.axis_mask = self.axis_mask.to(device)

    ##########################
    ### Deprecated Methods ###
    ##########################

    def optimizer_forward_pass(self, parameters, num_dipoles):
        """
        Perform an optimizer forward pass.

        This method takes a set of parameters and the number of dipoles, converts
        the parameter vector to individual parameters, and then performs a forward
        temporal pass using these parameters.

        Args:
            parameters (list or array-like): The parameter vector to be converted and used in the forward pass.
            num_dipoles (int): The number of dipoles to be considered in the forward pass.

        Returns:
            The result of the forward temporal pass.
        """
        warnings.warn(
            "optimizer_forward_pass is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.forward_temporal(
            *self.parametervector2parameters(parameters, num_dipoles)
        )

    def parametervector2parameters(self, parameters, num_dipoles):
        """
        Converts a parameter vector into separate dipole moment and position parameters.

        Parameters:
        -----------
        parameters : torch.Tensor or numpy.ndarray
            The input parameter vector. If it is a numpy array, it will be reshaped and converted.
        num_dipoles : int
            The number of dipoles.

        Returns:
        --------
        tuple
            A tuple containing:
            - m_dipoles: The dipole moments, scaled appropriately.
            - r_dipoles: The dipole positions, scaled appropriately.
        """
        warnings.warn(
            "parametervector2parameters is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(parameters, np.ndarray):
            parameters = np.array(parameters).reshape(
                -1, num_dipoles, self.dof_per_dipole
            )
            m_dipoles = parameters[:, :, :3] * self.m_scaling.detach().cpu().numpy()
            r_dipoles = parameters[:, :, 3:] * self.r_scaling.detach().cpu().numpy()
            return m_dipoles, r_dipoles

        parameters = parameters.view(-1, num_dipoles, self.dof_per_dipole)
        m_dipoles = parameters[:, :, :3] * self.m_scaling
        r_dipoles = parameters[:, :, 3:] * self.r_scaling
        return m_dipoles, r_dipoles

    def parameters2parametervector(
        self, m_dipoles, r_dipoles, as_numpy=False, determine_scaling=False
    ):
        """
        Converts dipole parameters into a parameter vector.

        Args:
            m_dipoles (torch.Tensor or array-like): Magnetic dipole moments.
            r_dipoles (torch.Tensor or array-like): Position vectors of the dipoles.
            as_numpy (bool, optional): If True, returns the parameter vector as a numpy array. Defaults to False.
            determine_scaling (bool, optional): If True, determines and applies scaling factors for the dipoles. Defaults to False.

        Returns:
            torch.Tensor or numpy.ndarray: Flattened parameter vector.
        """
        warnings.warn(
            "parameters2parametervector is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(m_dipoles, torch.Tensor):
            m_dipoles = torch.tensor(m_dipoles, dtype=torch.float32, device=self.device)
        if not isinstance(r_dipoles, torch.Tensor):
            r_dipoles = torch.tensor(r_dipoles, dtype=torch.float32, device=self.device)

        if determine_scaling:
            cat = torch.concat((m_dipoles, r_dipoles), dim=1)
            scale = torch.linalg.vector_norm(
                cat, ord=1, dim=(0, 2)
            ) / torch.linalg.vector_norm(cat, ord=1)
            self.m_scaling = scale[: m_dipoles.shape[1]].unsqueeze(-1)
            self.r_scaling = scale[m_dipoles.shape[1] :].unsqueeze(-1)

        parameters = torch.cat(
            (m_dipoles / self.m_scaling, r_dipoles / self.r_scaling), dim=-1
        )
        if as_numpy:
            return parameters.view(-1).detach().cpu().numpy()
        return parameters.view(-1)

    def bounds2parameterbounds(self, m_bnds, r_bnds, repeat=1):
        """
        Scales the given bounds by the scaling factors and repeats the bounds to match the number of time steps.
        The bounds are expected to be in the format of (min, max) pairs for each parameter.

        Parameters:
        -----------
        m_bnds : array-like
            The bounds for the m parameter. Should be convertible to a numpy array of dtype float32.
        r_bnds : array-like
            The bounds for the r parameter. Should be convertible to a numpy array of dtype float32.
        repeat : int, optional
            The number of times to repeat the bounds. Default is 1.

        Returns:
        --------
        numpy.ndarray
            A numpy array containing the scaled and repeated bounds, with shape (repeat, 2).
        """
        warnings.warn(
            "bounds2parameterbounds is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        m_bnds = np.array(m_bnds, dtype=np.float32)
        r_bnds = np.array(r_bnds, dtype=np.float32)
        m_scaling = self.m_scaling.unsqueeze(-1).detach().cpu().numpy()
        r_scaling = self.r_scaling.unsqueeze(-1).detach().cpu().numpy()
        scaled_m_bnds = m_bnds / m_scaling
        scaled_r_bnds = r_bnds / r_scaling
        return np.tile(
            np.concatenate((scaled_m_bnds, scaled_r_bnds), axis=1).reshape(-1, 2),
            (repeat, 1),
        )

    def plot_sensor_signals(self, field, time=None, mask_axis=True, **kwargs):
        """
        Plots the sensor signals for the given field data.

        Parameters:
        field (numpy.ndarray): The field data to be plotted. It is expected to be a 2D array.
        time (numpy.ndarray, optional): The time data corresponding to the field data. Default is None.
        mask_axis (bool, optional): If True, masks the field data where self.axis_mask is 0. Default is True.
        **kwargs: Additional keyword arguments to be passed to the plot_sensor_signals function.

        Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
        """
        warnings.warn(
            "plot_sensor_signals is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if mask_axis:
            field[:, self.axis_mask == 0] = float("nan")

        kwargs["field"] = field
        if time is not None:
            kwargs["time"] = time

        return plot_utils.plot_sensor_signals(**kwargs)

    def plot_field(self, field, savename=None):
        """
        Plots the magnetic field components (B_x, B_y, B_z) as 2D images.

        Parameters:
        field (numpy.ndarray): A 2D array where each column represents a magnetic field component (B_x, B_y, B_z).
        savename (str, optional): The name of the file to save the plot. If None, the plot is not saved. Default is None.
        """
        warnings.warn(
            "plot_field is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        field_components = [r"$B_x$", r"$B_y$", r"$B_z$"]
        # uT to nT
        field_ = field * 1e3
        fig, axes = plt.subplots(1, 3, figsize=(7.11, 2), dpi=500)
        sensor_x = np.unique(self.r_sensors[:, 0])
        sensor_z = np.unique(self.r_sensors[:, 2])

        for i, ax in enumerate(axes):
            c = ax.imshow(
                field_[:, i].reshape((len(sensor_z), len(sensor_x))),
                cmap="viridis",
                aspect="equal",
            )
            ax.set_title(field_components[i])
            if i == 0:
                ax.set_ylabel("Sensor Z [m]")
                ax.set_yticks(np.arange(len(sensor_z)))
                ax.set_yticklabels(np.round(sensor_z, 2))
            else:
                ax.set_yticks([])
            ax.set_xlabel("Sensor X [m]")
            ax.set_xticks(np.arange(len(sensor_x)))
            ax.set_xticklabels(np.round(sensor_x, 2))

        fig.colorbar(c, ax=axes.ravel().tolist(), label="Magnetic Field [nT]")
        if savename is not None:
            plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
        plt.show()


class CurrentDipoleForwardModel(ForwardModel):
    """Magnetic field of an **electric current dipole** in an infinite homogeneous
    medium (free space), via the Biot-Savart law.

    Drop-in replacement for :class:`ForwardModel` in the reconstruction pipeline:
    it exposes the same interface (``forward_linear`` / ``forward_temporal`` /
    ``forward_numpy``, same shapes, same axis-masking, linear in the dipole moment)
    but the heart is modelled as a current dipole with moment ``Q`` (A*m) rather
    than a magnetic dipole with moment ``m`` (A*m^2).

    Physics (``d = r_sensor - r_dipole``)::

        B = (mu0/4pi) Q x d / |d|^3          (Biot-Savart, ~1/r^2, Q x r_hat)
          = (mu0/4pi) (-[d]_x / |d|^3) Q      ([d]_x Q == d x Q)

    contrasted with the magnetic point dipole ``B = (mu0/4pi)(3(m.r_hat)r_hat - m)/r^3``
    (~1/r^3). The volume conductor is neglected (free space), which for the
    magnetic field is a mild approximation; the payoff is that the recovered
    moment trajectory is a more faithful VCG-like representation of the heart.
    """

    # Levi-Civita tensor: [d]_x[a, c] = _LEVI_CIVITA[a, b, c] * d[b], so that
    # [d]_x @ q == d x q. Cached per device (tiny, 27 entries).
    _LEVI_CIVITA = None

    @classmethod
    def _levi_civita(cls, device):
        eps = cls._LEVI_CIVITA
        if eps is None or eps.device != torch.device(device):
            eps = torch.zeros(3, 3, 3, device=device)
            for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                eps[a, b, c] = 1.0
            for a, b, c in ((0, 2, 1), (2, 1, 0), (1, 0, 2)):
                eps[a, b, c] = -1.0
            cls._LEVI_CIVITA = eps
        return eps

    def forward_linear(self, r_dipoles, m_dipoles=None, as_numpy=False, silent=False):
        """Biot-Savart lead field for a current dipole. Same contract as
        :meth:`ForwardModel.forward_linear`.

        If ``m_dipoles`` is None returns the field operator of shape
        ``(T, S*3, D*3)`` (axis-masked over sensor axes); otherwise returns the
        field ``(T, S, 3)`` (in uT) for moments ``m_dipoles``.
        """
        if not isinstance(r_dipoles, torch.Tensor) and not silent:
            r_dipoles = torch.tensor(
                r_dipoles.copy(), dtype=torch.float32, device=self.device
            )
            logging.debug(
                "Warning: not a tensor. Continuing without tracking gradients."
            )

        assert r_dipoles.shape[-1] == 3

        if r_dipoles.ndim == 1:
            r_dipoles = r_dipoles.reshape((1, 3))
        if r_dipoles.ndim == 2:
            r_dipoles = r_dipoles.reshape((1, -1, 3))

        if m_dipoles is not None:
            if not isinstance(m_dipoles, torch.Tensor) and not silent:
                m_dipoles = torch.tensor(
                    m_dipoles.copy(), dtype=torch.float32, device=self.device
                )
                logging.debug(
                    "Warning: not a tensor. Continuing without tracking gradients."
                )

            assert m_dipoles.shape[-1] == 3

            if m_dipoles.ndim == 1:
                m_dipoles = m_dipoles.reshape((1, 3))
            if m_dipoles.ndim == 2:
                m_dipoles = m_dipoles.reshape((1, -1, 3))

            assert m_dipoles.ndim == 3 and r_dipoles.ndim == 3
            assert m_dipoles.shape[-2] == r_dipoles.shape[-2]

            T, D, _ = m_dipoles.shape  # [T x D x 3]
            m_vec = m_dipoles.reshape(T, D * 3, 1)  # [T x D*3 x 1]

        # Displacement vectors d = r_sensor - r_dipole and their norms
        d = self.r_sensors.reshape(1, -1, 1, 3) - r_dipoles.reshape(
            r_dipoles.shape[0], 1, -1, 3
        )  # [T x S x D x 3]
        d_norm = torch.norm(d, dim=-1, keepdim=True)  # [T x S x D x 1]

        # Biot-Savart linear operator: block = mu0_4pi * (-[d]_x) / |d|^3.
        # Build [d]_x directly via the Levi-Civita contraction (single autograd
        # node, no stacked intermediates), keeping this as lean as the magnetic
        # ForwardModel: [d]_x[a, c] = eps_abc d_b, so [d]_x @ Q == d x Q.
        mu0_4pi = 0.1  # mu0 / (4 * pi); in uT | mu0 = 4pi * 10^-7 H/m
        eps = self._levi_civita(d.device)
        skew = torch.einsum("abc,tsdb->tsdac", eps, d)  # [T x S x D x 3 x 3]
        A = -mu0_4pi * skew / (d_norm**3).unsqueeze(-1)  # [T x S x D x 3 x 3]

        # Vectorized representation, identical layout to ForwardModel
        T, S, D, _, _ = A.shape
        A_mat = A.permute(0, 1, 3, 2, 4).reshape(T, S * 3, D * 3)  # [T x S*3 x D*3]

        if m_dipoles is None:
            if self.axis_mask is not None:
                A_mat = A_mat[:, self.axis_mask.flatten() == 1]
            if as_numpy:
                A_mat = A_mat.detach().cpu().numpy()
            return A_mat

        B_vec = torch.matmul(A_mat, m_vec)  # [T x S*3 x 1]
        B_total = B_vec.reshape(T, S, 3)  # [T x S x 3]
        if as_numpy:
            B_total = B_total.detach().cpu().numpy()
        return B_total

    def forward_temporal(self, m_dipoles, r_dipoles):
        """Compact Biot-Savart field over time (uT). Same shapes as
        :meth:`ForwardModel.forward_temporal`; produces the same result as
        ``forward_linear`` when moments are given.
        """
        if not isinstance(self.r_sensors, torch.Tensor):
            self.r_sensors = torch.tensor(
                self.r_sensors, dtype=torch.float32, device=self.device
            )
        if not isinstance(m_dipoles, torch.Tensor):
            m_dipoles = torch.tensor(
                m_dipoles.copy(), dtype=torch.float32, device=self.device
            )
            logging.debug(
                "Warning: not a tensor. Continuing without tracking gradients."
            )
        if not isinstance(r_dipoles, torch.Tensor):
            r_dipoles = torch.tensor(
                r_dipoles.copy(), dtype=torch.float32, device=self.device
            )
            logging.debug("Not a tensor. Continuing without tracking gradients.")

        assert m_dipoles.shape[-1] == 3 and r_dipoles.shape[-1] == 3

        if m_dipoles.ndim == 1:
            m_dipoles = m_dipoles.reshape((1, 3))
        if m_dipoles.ndim == 2:
            m_dipoles = m_dipoles.reshape((1, -1, 3))
        if r_dipoles.ndim == 1:
            r_dipoles = r_dipoles.reshape((1, 3))
        if r_dipoles.ndim == 2:
            r_dipoles = r_dipoles.reshape((1, -1, 3))

        assert m_dipoles.ndim == 3 and r_dipoles.ndim == 3
        assert m_dipoles.shape[-2] == r_dipoles.shape[-2]

        # d = r_sensor - r_dipole  [T x S x D x 3]
        d = self.r_sensors.reshape(1, -1, 1, 3) - r_dipoles.reshape(
            r_dipoles.shape[0], 1, -1, 3
        )
        m_dipoles = m_dipoles.reshape(m_dipoles.shape[0], 1, -1, 3)  # [T x 1 x D x 3]
        mu0_4pi = 0.1  # in uT
        d_norm = torch.norm(d, dim=-1, keepdim=True)  # [T x S x D x 1]
        # Biot-Savart: B = mu0_4pi * (Q x d) / |d|^3, summed over dipoles
        B = mu0_4pi * torch.cross(m_dipoles, d, dim=-1) / d_norm**3
        return torch.sum(B, dim=-2)  # [T x S x 3]
