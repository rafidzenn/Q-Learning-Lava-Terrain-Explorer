# RL-Hazard-Navigator: Autonomous Volcanic Terrain Exploration

## Project Overview
This project implements a Reinforcement Learning (RL) agent designed to autonomously navigate highly stochastic and hazardous environments. Modeled as a Markov Decision Process (MDP), the system utilizes a custom Q-Learning algorithm to balance exploration efficiency with strict risk mitigation. The agent operates within a simulated volcanic terrain, continuously calculating optimal policies to avoid dynamic hazards like lava flows, craters, and gas emissions while maximizing its pathfinding efficiency to the target destination.

## Key Features
* **Stochastic Environment Modeling:** Simulates real-world uncertainty where agent actions have probabilistic outcomes due to environmental interference.
* **Risk-Aware Pathfinding:** Prioritizes agent safety by assigning severe negative rewards to hazardous states (lava, gas), ensuring the optimal policy heavily penalizes dangerous routes.
* **Q-Learning Implementation:** Utilizes a model-free reinforcement learning algorithm to iteratively discover the optimal action-value function.
* **Modular Architecture:** Separates the environment configuration, the MDP formulation, and the learning algorithm for easy scaling and testing.

## Mathematical Foundation

The core of the navigation system is framed as a Markov Decision Process, defined by the tuple $\langle S, A, P, R, \gamma \rangle$:

* **$S$ (State Space):** The set of all possible grid coordinates in the volcanic terrain.
* **$A$ (Action Space):** The valid movements available to the agent (e.g., Up, Down, Left, Right).
* **$P$ (Transition Probability):** $P(s' | s, a)$, representing the probability of moving to state $s'$ given action $a$ in state $s$. This accounts for the stochastic nature of the terrain (e.g., slipping on loose volcanic rock).
* **$R$ (Reward Function):** $R(s, a)$, the immediate reward received after transitioning from state $s$ via action $a$. High penalties are applied for hazard encounters, with positive rewards for reaching the goal.
* **$\gamma$ (Discount Factor):** Determines the importance of future rewards, ensuring the agent seeks the most efficient path rather than wandering indefinitely.

### Q-Learning Update Rule
The agent updates its knowledge base using the temporal difference learning equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

*(Where $\alpha$ is the learning rate, controlling how much newly acquired information overrides old information).*

## Upcoming Enhancements (Roadmap)
* **Enhanced Exploration Strategy:** Transitioning from an $\epsilon$-greedy approach to Upper Confidence Bound (UCB) action selection to optimize the exploration-exploitation trade-off.
* **Advanced State Tracking:** Implementing a memory buffer and telemetry dashboard to visualize the agent's learning convergence and hazard-avoidance metrics over distinct training episodes.
* **Dynamic Hazards:** Introducing moving environmental threats (e.g., expanding lava flows) to transition the environment from stationary to non-stationary.

## Technologies Used
* **Language:** Python 3.x
* **Libraries:** NumPy (Matrix operations), Matplotlib/Seaborn (Visualization of policy maps)

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/RL-Hazard-Navigator.git](https://github.com/yourusername/RL-Hazard-Navigator.git)
