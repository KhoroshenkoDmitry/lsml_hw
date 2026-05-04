# https://arxiv.org/pdf/2211.17192
# В этом задании вам предстоит реализовать функцию resample(sample_id: int) -> resampled_id: int
# Она принимает sample_id, распределённые как draft_dist (q) и возвращает resampled_id
# resampled_id должны быть распределены как target_dist (p)
# При этом максимально возможное число resampled_id должно совпадать с входными sampled_id
#
# Для этого предлагается воспользоваться алгоритмом Speculative Sampling:
# Оставлять sampled_id с вероятностью p / q
# А если не оставили, то ресэмплировать из normalize(max(0, p - q))


import numpy as np


class SpeculativeSampler:
    def __init__(self, draft_dist: np.ndarray, target_dist: np.ndarray):
        """
        Parameters
        ----------
        draft_dist : np.ndarray
            1D probability vector q over the vocabulary (sums to 1).
        target_dist : np.ndarray
            1D probability vector p over the vocabulary (sums to 1).

        Store whatever you need as attributes (e.g. the normalized residual
        distribution used on rejection).
        """
        self.draft_dist = draft_dist
        self.target_dist = target_dist
        diff = np.maximum(0, target_dist - draft_dist)
        self.residual = diff / diff.sum()

    def resample(self, sample_id: int, rng: np.random.Generator) -> int:
        """
        Apply the speculative-sampling accept/reject step.

        Parameters
        ----------
        sample_id : int
            A token id that was drawn from the draft distribution q.

        Returns
        -------
        int
            Either `sample_id` (if accepted) or a resampled_id drawn from the
            residual distribution normalize(max(0, p - q)) (if rejected).

        Hint: accept with probability min(1, p[sample_id] / q[sample_id]).
        """
        prob : float = min(1, self.target_dist[sample_id] / self.draft_dist[sample_id])
        is_accepted : bool = rng.random() < prob
        if is_accepted:
            return sample_id
        else:
            return rng.choice(len(self.residual), p=self.residual)

def run_experiment(vocab_size: int = 10, num_samples: int = 200_000, seed: int = None):
    rng = np.random.default_rng(seed)

    draft_dist = rng.dirichlet(np.ones(vocab_size))
    target_dist = rng.dirichlet(np.ones(vocab_size))

    sampler = SpeculativeSampler(draft_dist, target_dist)

    draft_samples = rng.choice(vocab_size, size=num_samples, p=draft_dist)

    final_samples = np.empty(num_samples, dtype=np.int64)
    num_accepted = 0
    for i, x in enumerate(draft_samples):
        y = sampler.resample(x, rng)
        final_samples[i] = y
        num_accepted += y == x

    empirical_dist = np.bincount(final_samples, minlength=vocab_size) / num_samples

    theoretical_accept = np.minimum(draft_dist, target_dist).sum()
    empirical_accept = num_accepted / num_samples
    print(f"Theoretical acceptance rate: {theoretical_accept:.4f}")
    print(f"Empirical   acceptance rate: {empirical_accept:.4f}")
    if empirical_accept < theoretical_accept - 0.01:
        print('❌ Empirical acceptance too low')
    else:
        print('✅ Empirical acceptance is ok')

    max_abs_err = np.max(np.abs(empirical_dist - target_dist))
    print(f"Max |empirical - target|: {max_abs_err:.4f}")

    if max_abs_err > 0.01:
        print("❌ Resulting distribution doesn't match with target")
    else:
        print("✅ Resulting distribution is ok")


if __name__ == "__main__":
    run_experiment()