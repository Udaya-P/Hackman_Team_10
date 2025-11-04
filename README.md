# Hangman AI Agent using Hidden Markov Models (HMM) and Deep Reinforcement Learning (DQN)

### Machine Learning Hackathon — Team 10 (AIML-B)

**Team Members**
- PES2UG23AM072 — PIDAPA INDIRA  
- PES2UG23AM088 — SAI HEMANTH MEDICHERLA  
- PES2UG23AM089 — SAI JASWANTH AKULA  
- PES2UG23AM111 — UDAYA PRAGNA GANGAVARAM  

---

## Overview
This project presents an intelligent Hangman-playing agent that combines **probabilistic modeling (HMM)** with **reinforcement learning (DQN)**.  
While traditional Hangman players rely on intuition or letter frequency, our model learns to predict letters using statistical dependencies (via HMM) and adaptive decision-making (via DQN).

---

## Objective
To create an AI agent capable of accurately guessing hidden words by:
1. Learning letter patterns and dependencies using **Hidden Markov Models**.
2. Using **Deep Q-Learning** to develop optimal guessing strategies.
3. Managing the **exploration–exploitation** trade-off for efficient decision-making.

---

## Repository Contents

---

## Methodology

### 1. Hidden Markov Model (HMM)
- A **Per-Length HMM structure** was implemented, training separate HMMs for each word length to capture unique transition patterns.
- Used `MultinomialHMM` from `hmmlearn`, where each state emits discrete letter observations.
- Words are encoded as **one-hot vectors**, allowing the HMM to model letter-to-letter dependencies.
- Training is performed using the **Baum–Welch algorithm** (Expectation–Maximization), which internally applies the **forward and backward algorithms** to refine parameters.

**Model Components**
- **Transition Matrix (`transmat_`)** – represents how hidden letter states evolve across the word.
- **Emission Matrix (`emissionprob_`)** – defines the probability of observing each letter from each hidden state.
- Both matrices are initialized uniformly and iteratively updated to maximize corpus likelihood.

**Impact:**  
This per-length design improved both accuracy and efficiency by specializing HMMs for different word lengths, leading to better letter prediction and linguistic generalization.

---

### 2. Reinforcement Learning (DQN)
After HMM training, a **Deep Q-Network** was developed using PyTorch to learn decision-making policies for letter guessing.

**Environment Setup**
- The custom `HangmanEnv` class defines the game logic and rewards.
- State vector includes:
  1. Masked word encoding (revealed letters)
  2. Binary vector of guessed letters
  3. HMM-based letter probability vector
  4. Normalized remaining lives value

**Reward Function**
| Event | Reward |
|--------|--------|
| Correct guess | +2 |
| Word completed | +10 |
| Incorrect guess | -1 |
| Repeated guess | -3 |
| Lost all lives | -10 |

**DQN Architecture**
- Two hidden layers with 256 neurons each, using ReLU activation.
- Input dimension = (masked word + guessed + HMM + life features)
- Output dimension = 26 (one for each letter)
- Trained using **Adam optimizer**, **γ = 0.99**, and **learning rate = 1e-4**.

**Training Configuration**
| Parameter | Value |
|------------|-------|
| Episodes | 10,000 |
| Batch Size | 128 |
| Buffer Capacity | 150,000 |
| Target Update Frequency | 600 |
| Epsilon Decay | 0.9993 |
| Exploration Range | ε = 1.0 → 0.05 |

**Mechanics**
- **Experience Replay:** Stores (state, action, reward, next_state, done) tuples to reduce temporal correlation.
- **Target Network:** Stabilizes training by periodically syncing with the policy network.
- **Epsilon-Greedy Strategy:** Balances exploration and exploitation across training.

**Result:**  
The trained DQN learns to prioritize plausible letters based on HMM priors early in the game and adapt dynamically as more letters are revealed, outperforming random and frequency-based baselines.

---

## Exploration vs. Exploitation
The ε-greedy policy manages the exploration–exploitation trade-off:
- High ε (1.0) encourages exploration during early episodes.
- Gradual decay to ε = 0.05 favors exploitation of learned Q-values.
- Replay memory and target synchronization ensure stability and convergence.

---

## Results
The combined HMM + DQN framework achieved:
- Higher **success rates** compared to heuristic and random models.
- Lower **average wrong guesses per word**.
- Improved adaptability across word lengths and structures.

---

## Future Improvements
1. **Contextual HMM Smoothing** — Combine forward and backward probabilities for better handling of rare word patterns.  
2. **Enhanced DQN Variants** — Explore Double DQN, Dueling DQN, or Actor–Critic methods for improved policy optimization.  
3. **Curriculum Learning** — Start training with shorter words and scale up gradually.  
4. **Adaptive Reward Shaping** — Introduce rewards for partial progress (e.g., revealing vowels).  
5. **Hybrid Integration** — Feed HMM probabilities directly into the DQN as dynamic features.  
6. **Visualization Tools** — Build an interactive dashboard to visualize HMM transitions and DQN Q-value evolution.

---

## How to Run

### Requirements
Install dependencies:
```bash
pip install numpy torch torchvision hmmlearn tqdm
```

## Execution

1. Open **`ML_HACKATHON_TEAM10_AIML_B.ipynb`** in **Google Colab** or **Jupyter Notebook**.  
2. Upload **`corpus.txt`** when prompted.  
3. Run **all cells sequentially** to execute the workflow.  
   - The **Hidden Markov Models (HMMs)** will train first.  
   - The **Deep Q-Network (DQN)** training and evaluation will follow.  
4. Once training is complete, the following models will be saved automatically:
   - `per_length_hmms.pkl` – Trained HMM models for different word lengths.  
   - `dqn_hangman_policy.pth` – Trained DQN policy weights.  
5. These saved models can be **reloaded later for testing and evaluation** without retraining.

---

## Conclusion

This project successfully integrates **probabilistic reasoning (HMM)** and **reinforcement learning (DQN)** to build a robust, adaptive **Hangman-playing agent**.  
By leveraging **HMM-derived linguistic probabilities** and **DQN-based dynamic learning**, the system achieves:
- Efficient and context-aware letter prediction.  
- Improved accuracy over heuristic or random guessing approaches.  
- Continuous improvement through experience and reward-driven learning.

The result is a hybrid AI model capable of intelligent, data-driven gameplay in a language-based environment.

---

## License

This project was developed as part of the **Machine Learning Hackathon 2025 (AIML-B Section)**  
and is intended solely for **academic and research purposes**.

