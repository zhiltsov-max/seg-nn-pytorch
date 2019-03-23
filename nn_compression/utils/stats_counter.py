class StatsCounter:
    # Implements online parallel computation of sample variance
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    def pairwise_stats(self, count_a, avg_a, var_a, count_b, avg_b, var_b):
        delta = avg_b - avg_a
        m_a = var_a * (count_a - 1)
        m_b = var_b * (count_b - 1)
        M2 = m_a + m_b + delta ** 2 * count_a * count_b / (count_a + count_b)
        return [
            count_a + count_b,
            avg_a * 0.5 + avg_b * 0.5,
            M2 / (count_a + count_b - 1)
        ]

    # stats = float array of shape ( n, 2 * d ), d = dimensions of values
    # count = integer array of shape ( n )
    # mean_accessor = function(idx, stats) to retrieve element mean
    # variance_accessor = function(idx, stats) to retrieve element variance
    # Recursively computes total count, mean and variance, O(log(N)) calls
    def compute_stats(self, stats, counts, mean_accessor, variance_accessor):
        m = mean_accessor
        v = variance_accessor
        n = len(stats)
        if n == 1:
            return counts[0], m(0, stats), v(0, stats)
        if n == 2:
            return self.pairwise_stats(
                counts[0], m(0, stats), v(0, stats),
                counts[1], m(1, stats), v(1, stats)
                )
        h = n // 2
        return self.pairwise_stats(
            *self.compute_stats(stats[:h], counts[:h], m, v),
            *self.compute_stats(stats[h:], counts[h:], m, v)
            )