import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst

# K-armed Bernoulli bandit
# means: List of reward means
# strat: epsilon-greedy (0) or UCB (1)
# M rounds of simulation, each for T timesteps

def bernoulli_bandit(means, strat, param, M, T):
    best = np.argmax(means)
    K = len(means)

    regrets = np.zeros((M,T))
    bestArmFreq = np.zeros((M,T))
    bestArmQ = np.zeros((M,T))

    for m in range(M):
        # Keep track of action values and action counts
        Q = np.zeros(K)
        N = np.zeros(K)

        for t in range(T):
            # Action selection depending on strategy
            if strat == 0:
                if np.random.random() < param:
                    arm = np.random.randint(0, K)
                else: 
                    arm = np.argmax(Q)
            else:                
                arm = np.argmax(Q+param*np.sqrt(np.log(t+1)/(N+1)))

            # Reward and Q value update
            reward = np.random.binomial(1, means[arm])
            N[arm] += 1
            Q[arm] += 1/N[arm]*(reward-Q[arm])

            # Track frequency of choosing best arm and actual regret
            if arm == best:
                regrets[m,t] = 0
            else:
                regrets[m,t] = np.random.binomial(1,means[best])-reward
            bestArmFreq[m,t] = N[best]/(t+1)
            bestArmQ[m,t] = Q[best]

    return np.mean(regrets,axis=0), np.mean(bestArmFreq,axis=0)


def execute(means, strat, params=[0.1, 0.2, 0.3], M=100, T=10000):
    f = plt.figure()
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212, sharex=ax1)

    for p in params:
        regrets, bestArmFreq = bernoulli_bandit(means, strat, p, M, T)

        # Plot cumulative regret
        cum_regrets = np.cumsum(regrets)
        if strat == 0:
            ax1.plot(cum_regrets, label=f"e={p:.2f}")
        else:
            ax1.plot(cum_regrets, label=f"c={p:.2f}")

        # Plot best arm frequency
        ax2.plot(bestArmFreq)

        # Print final values
        final_cum_regret = cum_regrets[-1]
        final_pct_best_arm = bestArmFreq[-1]
        print(
            f"Param = {p:.2f} | Final Cumulative Regret: {final_cum_regret:.2f} | "
            f"Final % Best Arm Played: {final_pct_best_arm:.2f}"
        )

    ax1.set_xscale("log", base=10)
    ax1.set_title("cumulative regret")
    ax2.set_title("percentage best arm played")
    f.legend()
    f.tight_layout()
    plt.show()


execute(means=[0.2, 0.4, 0.6, 0.8], strat=0)
