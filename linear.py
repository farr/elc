import numpy as np

def solve_linear_gaussian_model(M, y):
    r"""Returns the mean and covariance matrix for the Gaussian likelihood
    on the vector of parameters :math:`\vec{x}` in a linear model for
    :math:`\vec{y}` with response matrix :math:`\mathbf{M}`:

    .. math::

      \vec{y} = \mathbf{M} \vec{x} + \vec{\epsilon}

    with the error term :math:`\vec{\epsilon} \sim N(0,1)`.

    :return: ``(mu_x, Sigma_x)``, the mean and covariance matrix for
      the posterior on the parameters ``x``.

    """

    U, S, V = np.linalg.svd(M, full_matrices=False)

    Sigma = np.dot(V.T, np.dot(np.diag(1/(S*S)), V))
    
    mu = np.dot(np.dot(V.T, np.dot(np.diag(1/S), U.T)), y)

    return (mu, Sigma)
