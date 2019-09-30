"""
This module contains functions to perform power iteration with deflation
to compute the top eigenvalues and eigenvectors of a linear operator
"""
import numpy as np
import torch


class Operator(object):
    """
    maps x -> Lx for a linear operator L
    """

    def __init__(self, size):
        self.size = size

    def apply(self, vec):
        """
        Function mapping vec -> L vec where L is a linear operator
        """
        raise NotImplementedError


class LambdaOperator(Operator):
    """
    Linear operator based on a provided lambda function
    """
    def __init__(self, apply_fn, size):
        super(LambdaOperator, self).__init__(size)
        self.apply_fn = apply_fn

    def apply(self, x):
        return self.apply_fn(x)


def deflated_power_iteration(operator,
                             num_eigenthings=10,
                             power_iter_steps=20,
                             power_iter_err_threshold=1e-4,
                             momentum=0.0,
                             device=torch.device('cpu')):
    """
    Compute top k eigenvalues by repeatedly subtracting out dyads
    operator: linear operator that gives us access to matrix vector product
    num_eigenvals number of eigenvalues to compute
    power_iter_steps: number of steps per run of power iteration
    power_iter_err_threshold: early stopping threshold for power iteration
    returns: np.ndarray of top eigenvalues, np.ndarray of top eigenvectors
    """
    eigenvals = []
    eigenvecs = []
    current_op = operator

    def _deflate(x, val, vec):
        return val * vec.dot(x) * vec
    counter = 0
    for _ in range(num_eigenthings):
        counter += 1
        eigenval, eigenvec = power_iteration(current_op, power_iter_steps,
                                             power_iter_err_threshold,
                                             momentum=momentum,
                                             device=device)

        def _new_op_fn(x, op=current_op, val=eigenval, vec=eigenvec):
            return op.apply(x) - _deflate(x, val, vec)
        current_op = LambdaOperator(_new_op_fn, operator.size)
        eigenvals.append(eigenval)
        eigenvecs.append(eigenvec.cpu())
        print('Eigenvalue ' + str(counter) + ' computed.  Ans = ' + str(eigenval))
    return eigenvals, eigenvecs


def power_iteration(operator, steps=20, error_threshold=1e-4, dtype=torch.float,
                    momentum=0.0, device=torch.device('cpu'), offset=0, quiet=False):
    """
    Compute dominant eigenvalue/eigenvector of a matrix
    operator: linear Operator giving us matrix-vector product access
    steps: number of update steps to take
    returns: (principal eigenvalue, principal eigenvector) pair
    """
    if momentum > 0:
        raise ValueError('Momentum is disabled.')
    vector_size = operator.size  # input dimension of operator
    vec = torch.randn(vector_size, device=device, dtype=dtype)
    vec /= vec.norm()

    prev_lambda = 0.
    lambda_estimate = 0.
    # prev_vec = torch.zeros_like(vec)
    for step in range(steps):
        new_vec = operator.apply(vec).detach()  # - momentum * prev_vec
        # prev_vec = vec / torch.norm(vec)

        lambda_estimate = vec.dot(new_vec).item()
        vec = new_vec / torch.norm(new_vec)

        diff = lambda_estimate - prev_lambda
        error = np.abs(diff)
        if error < error_threshold * np.abs(lambda_estimate):
            if lambda_estimate < 1 or error < error_threshold:
                # EV certificate
                if not quiet:
                    residual = (new_vec - lambda_estimate * vec)
                    l2_cert = residual.norm().item()
                    linf_cert = residual.abs().max().item()
                    print(f'Current estimate is {lambda_estimate + offset:.6} in step {step},'
                          f' with rel. tol {error/np.abs(lambda_estimate):.6} and'
                          f' cert [2,inf]: [{l2_cert:.6},{linf_cert:.6}].')
                return lambda_estimate, vec
        prev_lambda = lambda_estimate

        if step % 100 == 0:
            # EV certificate
            residual = (new_vec - lambda_estimate * vec)
            l2_cert = residual.norm().item()
            linf_cert = residual.abs().max().item()
            print(f'Current estimate is {lambda_estimate + offset:.6} in step {step},'
                  f' with rel. tol {error/np.abs(lambda_estimate):.6} and'
                  f' cert [2,inf]: [{l2_cert:.6},{linf_cert:.6}].')

    return lambda_estimate, vec


def smallest_eigenvalue(operator,
                        power_iter_steps=20,
                        power_iter_err_threshold=1e-4,
                        momentum=0.0,
                        device=torch.device('cpu'), dtype=torch.float,
                        quiet=False):
    """Compute top eigenvalue in magnitude first and then subtract it to get the lowest eigenvalue in absolute value.

    Look at
    A = H - |lambda|I
    operator: linear operator that gives us access to matrix vector product
    num_eigenvals number of eigenvalues to compute
    power_iter_steps: number of steps per run of power iteration
    power_iter_err_threshold: early stopping threshold for power iteration
    returns: np.ndarray of top eigenvalues, np.ndarray of top eigenvectors
    """
    max_eigenval, _ = power_iteration(operator, power_iter_steps,
                                      power_iter_err_threshold,
                                      momentum=momentum,
                                      device=device,
                                      dtype=dtype,
                                      offset=0, quiet=quiet)
    print('Max abs. EV computed.  Ans = ' + str(max_eigenval))

    def _deflate(x, val, vec):
        return val * vec.dot(x) * vec

    def _new_op_fn(x):
        return operator.apply(x) - np.abs(max_eigenval) * x

    new_operator = LambdaOperator(_new_op_fn, operator.size)

    smallest_ev, _ = power_iteration(new_operator, power_iter_steps,
                                     power_iter_err_threshold,
                                     momentum=momentum,
                                     device=device,
                                     dtype=dtype,
                                     offset=np.abs(max_eigenval))
    smallest_ev += np.abs(max_eigenval)
    print('Min total EV computed.  Ans = ' + str(smallest_ev))
    return max_eigenval, smallest_ev
