"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# PART 1 (20 points), Q1: Show computations for the given A and B.
# The returned products evaluate to AB=[[3,2],[2,3]], BA=[[7,6],[-2,-1]],
# (A+B)^T=[[4,2],[6,0]], AB^T=[[11,-4],[4,-1]], trace(A)=4, trace(AB)=6.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.array([[3., 4.], [2., 1.]]) if A is None else np.asarray(A, dtype=float)
    B = np.array([[1., 2.], [0., -1.]]) if B is None else np.asarray(B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(AB)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# det(M-lambda I)=(lambda-1)(lambda-3). Unit eigenvectors may be chosen as
# (1,1)^T/sqrt(2) and (1,-1)^T/sqrt(2); eigh returns an equivalent basis.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    M = np.array([[2., -1.], [-1., 2.]]) if M is None else np.asarray(M, dtype=float)
    return np.linalg.eigh(M)


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# trace(AB)=sum_i sum_j A_ij B_ji = sum_j sum_i B_ji A_ij = trace(BA).

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# det(C)=10, so C is invertible and C^-1=(1/10)[[6,-7],[-2,4]].
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.array([[4., 7.], [2., 6.]]) if C is None else np.asarray(C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# Normalization identifies a Poisson(rate) law: c=exp(-rate), E[X]=Var(X)=rate.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    if rate <= 0:
        raise ValueError('rate must be positive')
    return np.exp(-rate), float(rate), float(rate)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# Use P(A intersect B)=P(A|B)P(B), Bayes' rule, and inclusion-exclusion.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    intersection = p_a_given_b * p_b
    return intersection, intersection / p_a, p_a + p_b - intersection


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# The likelihood is theta^heads(1-theta)^(tosses-heads), maximized at heads/tosses.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    if tosses <= 0 or not 0 <= heads <= tosses:
        raise ValueError('heads must be in [0, tosses], with tosses positive')
    mle = heads / tosses
    # Posterior is proportional to theta**(heads+1)*(1-theta)**tails.
    # Its interior mode is (heads+1)/(tosses+1); it also handles zero heads.
    map_estimate = (heads + 1) / (tosses + 1)
    return mle, map_estimate


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# Gaussian MLEs are the sample mean and the mean squared deviation (divisor n).
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    x = np.array([1., 3., 5., 7.]) if data is None else np.asarray(data, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError('data must be a one-dimensional sample of length at least two')
    mean = np.mean(x)
    return mean, np.mean((x - mean) ** 2)


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# Posterior precision is the sum of likelihood and prior precisions; its mean is the MAP.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    if observation_variance <= 0 or prior_variance <= 0:
        raise ValueError('variances must be positive')
    precision = 1 / observation_variance + 1 / prior_variance
    return (x / observation_variance + prior_mean / prior_variance) / precision


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# Expand E[(X-E[X])^2] and use linearity of expectation to obtain E[X^2]-E[X]^2.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# With Sigma=E[(X-mu)(X-mu)^T], move fixed u outside E to get u^T Sigma u.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# A positive off-diagonal covariance creates the upward-slanting correlated cloud.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    if n_samples < 2:
        raise ValueError('n_samples must be at least 2')
    mean = np.array([2., -1.]) if mean is None else np.asarray(mean, dtype=float)
    covariance = (np.array([[3., 1.2], [1.2, 1.]]) if covariance is None
                  else np.asarray(covariance, dtype=float))
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError('mean and covariance must have shapes (2,) and (2,2)')
    return np.random.default_rng(seed).multivariate_normal(mean, covariance, n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.``
# Center X, form S=Xc.T@Xc/(n-1), sort its eigenpairs, then project Xc onto them.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 2 or not np.all(np.isfinite(X)):
        raise ValueError('X must be finite with shape (n, 2), n >= 2')
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (len(X) - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return {'mean': mean, 'centered': centered, 'covariance': covariance,
            'eigenvalues': values, 'components': vectors,
            'scores': centered @ vectors}


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# Divide each eigenvalue by their sum; PC1's fraction is retained by 1D PCA.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    total = eigenvalues.sum()
    if np.any(eigenvalues < 0) or total <= 0:
        raise ValueError('eigenvalues must be nonnegative with positive sum')
    return eigenvalues / total


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# The left panel shows samples and principal directions; the right panel shows variance fractions.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    X = np.asarray(X, dtype=float)
    mean, values, components = result['mean'], result['eigenvalues'], result['components']
    fractions = explained_variance(values)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(X[:, 0], X[:, 1], alpha=.65, label='samples')
    for i, color in enumerate(('tab:red', 'tab:green')):
        direction = components[:, i] * np.sqrt(values[i]) * 2
        axes[0].arrow(*mean, *direction, color=color, width=.025,
                      length_includes_head=True, label=f'PC{i + 1}')
    axes[0].scatter(*mean, c='black', marker='x', s=70, label='mean')
    axes[0].set(xlabel='Feature 1', ylabel='Feature 2', title='Data and PCA directions')
    axes[0].legend()
    axes[1].bar(['PC1', 'PC2'], fractions, color=['tab:red', 'tab:green'])
    axes[1].set(ylim=(0, 1), ylabel='Explained variance fraction', title='Explained variance')
    for i, fraction in enumerate(fractions):
        axes[1].text(i, fraction + .025, f'{fraction:.1%}', ha='center')
    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used. Used: https://chatgpt.com/s/cx_6ac8806e21d08191be97675a2142c0ae
# Contributions: completed derivations, NumPy PCA, plotting, and write-up. Resources: NumPy/Matplotlib docs.

if __name__ == '__main__':
    data = generate_data()
    fitted = pca(data)
    figure = make_plots(data, fitted)
    figure.savefig(Path(__file__).with_name('pca_results.png'), dpi=160, bbox_inches='tight')