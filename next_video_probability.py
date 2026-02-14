"""
Mathematical calculation module for next video's matching probability.

This module implements a Bayesian logistic model for predicting match probabilities
based on verification history across multiple ranked lists.

The model uses:
- Rank-to-score conversion (1/k)
- MAP estimation with Gaussian priors
- Laplace approximation for posterior uncertainty
- MacKay approximation for predictive probabilities
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid


def rank_to_score(k):
    """
    Convert rank k to score t_k = 1/k
    This ensures higher ranks (lower k) have higher scores.
    
    Args:
        k: rank position (1-indexed)
    
    Returns:
        float: score t_k = 1/k
    """
    return 1.0 / k


def compute_map_estimate(verification_history, sigma_square=100.0):
    """
    Compute MAP estimate of theta = [a, b] using BFGS optimization.
    
    The logistic model is:
        P(Y=1|rank) = σ(a + b * t_rank)
    where t_rank = 1/rank and σ is the sigmoid function.
    
    Args:
        verification_history: list of (rank, match) tuples where match ∈ {0, 1}
        sigma_square: variance for the Gaussian prior (σ_a² = σ_b² = sigma_square)
    
    Returns:
        theta_hat: MAP estimate [a, b]
        H_inv: inverse Hessian (covariance of Laplace approximation)
    """
    if len(verification_history) == 0:
        # No data yet, return prior mean and prior covariance
        return np.array([0.0, 0.0]), np.array([[sigma_square, 0], [0, sigma_square]])
    
    # Extract data
    ranks = np.array([vh[0] for vh in verification_history])
    y = np.array([vh[1] for vh in verification_history])
    t = np.array([rank_to_score(r) for r in ranks])
    
    n = len(verification_history)
    
    # Prior parameters
    mu_0 = np.array([0.0, 0.0])
    Sigma_0_inv = np.array([[1.0/sigma_square, 0], [0, 1.0/sigma_square]])
    
    def objective(theta):
        """Negative log-posterior J(θ)"""
        a, b = theta
        x = a + b * t  # linear predictor for all data points
        p = sigmoid(x)
        
        # Negative log-likelihood
        # Clip probabilities to avoid log(0)
        p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
        nll = -np.sum(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
        
        # Prior term
        diff = theta - mu_0
        prior_term = 0.5 * diff @ Sigma_0_inv @ diff
        
        return nll + prior_term
    
    def gradient(theta):
        """Gradient of J(θ)"""
        a, b = theta
        x = a + b * t
        p = sigmoid(x)
        
        # Gradient from likelihood
        residuals = p - y
        grad_a = np.sum(residuals)
        grad_b = np.sum(residuals * t)
        
        # Gradient from prior
        diff = theta - mu_0
        grad_prior = Sigma_0_inv @ diff
        
        return np.array([grad_a, grad_b]) + grad_prior
    
    def hessian(theta):
        """Hessian of J(θ)"""
        a, b = theta
        x = a + b * t
        p = sigmoid(x)
        
        # Hessian from likelihood
        w = p * (1 - p)
        H_aa = np.sum(w)
        H_ab = np.sum(w * t)
        H_bb = np.sum(w * t * t)
        
        H_likelihood = np.array([[H_aa, H_ab], [H_ab, H_bb]])
        
        return H_likelihood + Sigma_0_inv
    
    # Optimize using BFGS
    result = minimize(objective, x0=np.array([0.0, 0.0]), method='BFGS', jac=gradient)
    theta_hat = result.x
    
    # Compute Hessian at MAP estimate for Laplace approximation
    H = hessian(theta_hat)
    
    # Compute inverse Hessian (covariance of posterior)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        # If singular, add small regularization
        H_inv = np.linalg.inv(H + 1e-6 * np.eye(2))
    
    return theta_hat, H_inv


def compute_predictive_probability(next_rank, theta_hat, H_inv):
    """
    Compute predictive probability P(Y_{n+1}=1 | y_{1:n}) using MacKay approximation.
    
    This integrates over the posterior uncertainty in θ to get a robust prediction.
    The MacKay approximation accounts for model uncertainty through the Laplace
    approximation of the posterior.
    
    Args:
        next_rank: the rank of the next video to verify (1-indexed)
        theta_hat: MAP estimate [a, b]
        H_inv: inverse Hessian from Laplace approximation (posterior covariance)
    
    Returns:
        prob: predicted probability of match for the next video
    """
    t_next = rank_to_score(next_rank)
    C_next = np.array([1.0, t_next])
    
    # Mean and variance of the linear predictor
    m = C_next @ theta_hat
    s_squared = C_next @ H_inv @ C_next
    
    # MacKay approximation
    # P(Y_{n+1}=1) ≈ σ(m / sqrt(1 + π/8 * s²))
    scaling = np.sqrt(1 + np.pi / 8 * s_squared)
    prob = sigmoid(m / scaling)
    
    return prob


def compute_all_predictive_probabilities(ranked_lists_state, sigma_square=100.0):
    """
    Compute predictive probabilities for the next video in each ranked list.
    
    This is the main interface function that processes all ranked lists and
    returns the predictions sorted by probability, allowing the system to
    select the most promising video to verify next.
    
    Args:
        ranked_lists_state: list of dicts, each containing:
            - 'video_ids': list of video IDs in ranked order
            - 'pointer': current position (how many videos processed)
            - 'verification_history': list of (rank, match) tuples
        sigma_square: variance for Gaussian prior (default: 100.0)
    
    Returns:
        probabilities: list of (list_index, probability, next_video_id) tuples
                      sorted by probability in descending order
    """
    probabilities = []
    
    for i, state in enumerate(ranked_lists_state):
        pointer = state['pointer']
        video_ids = state['video_ids']
        
        # Check if this list has more videos to verify
        if pointer >= len(video_ids):
            continue
        
        # Next rank is pointer + 1 (1-indexed)
        next_rank = pointer + 1
        next_video_id = video_ids[pointer]
        
        # Compute MAP estimate and predictive probability
        theta_hat, H_inv = compute_map_estimate(state['verification_history'], sigma_square)
        prob = compute_predictive_probability(next_rank, theta_hat, H_inv)
        
        probabilities.append((i, prob, next_video_id))
    
    # Sort by probability descending
    probabilities.sort(key=lambda x: x[1], reverse=True)
    
    return probabilities
