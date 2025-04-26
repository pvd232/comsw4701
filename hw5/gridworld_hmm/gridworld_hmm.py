import numpy as np
import numpy.typing as npt

class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: list = [], num_particles: int = 30):
        if walls:
            self.grid = np.ones(size)
            for cell in walls:
                self.grid[cell] = 0
        else:
            self.grid = np.random.randint(2, size=size)

        self.init = (self.grid / np.sum(self.grid)).flatten()

        self.epsilon = epsilon
        self.particles = np.random.choice(len(self.init), size=num_particles, p=self.init)
        self.weights = np.ones(num_particles)

        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        m, n = self.grid.shape
        adjacent = [(i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
        neighbors = [(i, j)]
        for a1, a2 in adjacent:
            if 0 <= a1 < m and 0 <= a2 < n and self.grid[a1, a2] == 1:
                neighbors.append((a1, a2))
        return neighbors

    """
    4.1 and 4.2. Transition and observation probabilities
    """

    def initT(self)-> np.ndarray:
        """
        Create and return NxN transition matrix, where N = size of grid.
        """
        rows, cols = self.grid.shape
        N          = self.grid.size
        T          = np.zeros((N, N), dtype=float)

        for r in range(rows):
            for c in range(cols):
                i = r * cols + c                          # flattened index of (r, c)
                neigh = self.neighbors((r, c))            # includes (r, c) itself
                p = 1.0 / len(neigh)                      # uniform over S(x)

                for nr, nc in neigh:
                    j = nr * cols + nc                    # flattened neighbour index
                    T[i, j] = p

                T[i] /= T[i].sum()

        return T

    def initO(self):
        """
        Create and return 16xN matrix of observation probabilities, where N = size of grid.
        """
        rows, cols = self.grid.shape
        N = self.grid.size
        O = np.zeros((16, N), dtype=float)

        for r in range(rows):
            for c in range(cols):
                s = r * cols + c

                north = 1 if r-1 < 0            or self.grid[r-1, c] == 0 else 0
                east  = 1 if c+1 >= cols        or self.grid[r,   c+1] == 0 else 0
                south = 1 if r+1 >= rows        or self.grid[r+1, c] == 0 else 0
                west  = 1 if c-1 < 0            or self.grid[r,   c-1] == 0 else 0
                e_star = (north << 3) | (east << 2) | (south << 1) | west

                for e in range(16):
                    d = bin(e ^ e_star).count("1")               # Hamming distance
                    O[e, s] = (1 - self.epsilon) ** (4 - d) * (self.epsilon ** d)

                O[:, s] /= O[:, s].sum()

        return O

    """
    4.3. Forward algorithm
    """

    def forward(self, observations: list[int]):
        """Perform forward algorithm over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Estimated belief state at each timestep.
        """
        T_steps = len(observations)
        N = self.grid.size
        beliefs = np.zeros((T_steps, N), dtype=float)

        bel_prev = self.init.copy()                     
        for t, obs in enumerate(observations):

            bel_pred = bel_prev @ self.trans              

            bel = bel_pred * self.obs[obs]                
            s   = bel.sum()
            bel = bel / s if s > 0 else np.ones(N) / N    

            beliefs[t] = bel
            bel_prev   = bel

        return beliefs

    """
    4.4. Particle filter
    """

    def transition(self):
        """
        Sample the transition matrix for all particles.
        Update self.particles in place.
        """
        N = self.trans.shape[1]
        for i, state in enumerate(self.particles):
            self.particles[i] = np.random.choice(N, p=self.trans[state])

    def observe(self, observation):
        """
        Compute the weights for all particles.
        Update self.weights in place.
        Args:
          obs (int): Integer observation value.
        """
        self.weights = self.obs[observation, self.particles]
        total = self.weights.sum()
        if total == 0:                       # unlikely but guard anyway
            self.weights.fill(1.0 / len(self.weights))
        else:
            self.weights /= total

    def resample(self):
        """
        Resample all particles.
        Update self.particles and self.weights in place.
        """
        idx = np.random.choice(
            len(self.particles), size=len(self.particles), replace=True, p=self.weights
        )
        self.particles = self.particles[idx]
        self.weights.fill(1.0)  # reset to uniform

    def particle_filter(self, observations: list[int]):
        """Apply particle filter over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Counts of particles in each state at each timestep.
        """
        T_steps = len(observations)
        N = self.grid.size
        counts = np.zeros((T_steps, N), dtype=int)

        for t, obs in enumerate(observations):
            self.transition()  # sample motion model
            self.observe(obs)  # weight by sensor model
            self.resample()  # importance resampling
            # record counts (approximate posterior)
            counts[t] = np.bincount(self.particles, minlength=N)

        return counts
