"""
This module defines a linear operator to compute the hessian-vector product
for a given pytorch model using subsampled data.
"""
import torch
from .power_iter import Operator, deflated_power_iteration, smallest_eigenvalue
from scipy.sparse.linalg import LinearOperator, eigsh
import scipy
import numpy as np


class HVPOperatorFullSet(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    device: which device stuff is running on
    """

    def __init__(self, model, dataloader, criterion, device=torch.device('cpu'),
                 weight_decay=0, dtype=torch.float):
        size = int(sum(p.numel() for p in model.parameters()))
        super().__init__(size)
        self.model = model.to(device=device, dtype=dtype)
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.dtype = dtype
        self.reg = weight_decay

    def apply(self, vec):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        vec_seq = []
        loc = 0
        for param in self.model.parameters():
            vec_seq.append(vec[loc:loc + param.numel()].reshape_as(param))
            loc += param.numel()

        self.model.zero_grad()
        hvp = torch.zeros(self.size, device=self.device, dtype=self.dtype)
        for i, (inputs, targets) in enumerate(self.dataloader):
            inputs, targets = inputs.to(device=self.device, dtype=self.dtype), targets.to(self.device)

            loss = self.criterion(self.model(inputs), targets)
            # if self.reg > 0:
            #    for param in self.model.parameters():
            #        loss = loss + self.reg * 0.5 * param.pow(2).sum()
            grad_seq = torch.autograd.grad(loss, self.model.parameters(),
                                           only_inputs=True, create_graph=True, retain_graph=True)
            hvp_seq = torch.autograd.grad(grad_seq, self.model.parameters(), grad_outputs=vec_seq,
                                          only_inputs=True, retain_graph=False)
            hvp += torch.cat([hv.contiguous().view(-1) for hv in hvp_seq])
            hvp += self.reg * vec
        return hvp / (i + 1)

    def np_apply(self, vec):
        vec_torch = self.apply(torch.tensor(vec, dtype=self.dtype, device=self.device))
        return vec_torch.cpu().numpy()


class HessianOperator(LinearOperator):
    """
        Proper SciPy wrapper
    """
    def __init__(self, model, dataloader, criterion, device=torch.device('cpu'),
                 weight_decay=0, dtype=torch.float, vec_dtype='float32'):
        self.N = sum([p.numel() for p in model.parameters()])
        # LinOp:
        self.shape = (self.N, self.N)
        self.dtype = np.dtype(vec_dtype)

        # TorchOp
        self.device = device
        self.hvp_op = HVPOperatorFullSet(model, dataloader, criterion, device, weight_decay, dtype)

        # Operator Shift
        self.shift = 0

    def _matvec(self, np_vec):
        np_vec_float = self.hvp_op.np_apply(np_vec.flatten())
        return np_vec_float.astype(self.dtype) - self.shift * np_vec

    def _adjoint(self, np_vec):
        return self._matvec(np_vec)

    def _matmul(self, mat):
        raise ValueError('Too expensive!')


def eigenvalue_analysis(operator, method='power_method', tol=1e-6, max_iter=100, quiet=False):
    """Return largest EV in magnitude and smallest algebraic eigenvalue."""
    if method == 'power_method':
        eigmax, eigmin = smallest_eigenvalue(operator.hvp_op,
                                       power_iter_steps=max_iter,
                                       power_iter_err_threshold=tol,
                                       momentum=0.0,
                                       device=operator.device, quiet=quiet)
        return eigmax, eigmin
    elif method == 'lanczos':
        # Largest EV
        try:
            eigvals, eigvec = eigsh(operator, k=1, tol=tol, maxiter=max_iter)
        except Exception as e:
            return None
        maxeig = eigvals[0]
        residual = operator(eigvec.flatten()) - eigvals[0] * eigvec.flatten()
        cert_l2 = np.linalg.norm(residual)
        cert_linf = np.max(np.abs(residual))
        print(f'Max eigenvalue = {maxeig:f}, cert [2, inf]= [{cert_l2},{cert_linf}]')

        # Smallest EV
        operator.shift = maxeig
        try:
            eigvals, eigvec = eigsh(operator, k=1, tol=tol, maxiter=max_iter)
        except Exception as e:
            return None
        eigvals = eigvals + operator.shift
        mineig = eigvals[0]

        residual = operator(eigvec.flatten()) - eigvals[0] * eigvec.flatten()
        cert_l2 = np.linalg.norm(residual)
        cert_linf = np.max(np.abs(residual))
        print(f'Min eigenvalue = {mineig:f}, cert [2, inf]= [{cert_l2},{cert_linf}]')
        return (maxeig, mineig)
