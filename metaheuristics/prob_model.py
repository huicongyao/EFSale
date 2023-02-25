class ProbabilityModel:
    """
        Probabilistic-based model for Knowledge Transfer between tasks

        Class Parameters
        ----------
        prob_vec : numpy.ndarray
            Probability vector for Univariate Marginal Frequency

        noise_prob_vec : numpy.ndarray
            Noisy probability vector for Univariate Marginal Frequency
    """

    def __init__(self, prob_vec, noise_prob_vec):
        self.prob_vec = prob_vec
        self.noise_prob_vec = noise_prob_vec
