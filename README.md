# Deep Bluffing: Assignment 1

You will build two core algorithms: Value Iteration and Policy Iteration. You have been provided with the Python starter files. Your task is to complete the mathematical updates within the `TODO` blocks.
For reference: https://gibberblot.github.io/rl-notes/index.html

> **Note:** Do not modify the function signatures or the boilerplate environment setup. Focus solely on translating the Bellman equations into Python logic.

---

## Task 1: Value Iteration (Model-Based)

**File:** `value_iteration.py`

Value Iteration is a dynamic programming algorithm that computes the optimal value function $v_*(s)$ assuming we have perfect knowledge of the environment's transition probabilities $\mathcal{P}_{ss'}^a$ and rewards $\mathcal{R}_s^a$.

The algorithm iteratively updates the value of each state using the Bellman Optimality Equation until the maximum change across all states is less than a threshold $\theta$:


$$v_{\*}(s)=\max_{a}\left(\mathcal{R}_{s}^{a}+\gamma\sum_{s'\in S}\mathcal{P}_{ss'}^{a}v_{\*}(s')\right)$$

**Your Task:**
Locate the nested loops in `value_iteration.py`.

1. Compute the expected return (Q-value) for taking a specific action $a$ in state $s$.
2. Update the new value of state $s$ to be the maximum of these Q-values.

---

## Task 2: Policy Iteration (Model-Based)

**File:** `policy_iteration.py`

While Value Iteration iterates over values, Policy Iteration directly updates the policy itself. It consists of two repeating steps: evaluating the current policy, and strictly improving it.

**Your Task:**

1. **Policy Evaluation:** Calculate the expected value of following the *current* policy's chosen action.
2. **Policy Improvement:** For each state, check if there is an action that yields a higher expected return than the current policy's action. If so, update the policy to choose the new $\arg\max_a$ action.

---

## Task 3: Watch Your Algorithms Play!

Once you have completed the `TODO` blocks and your math is correct, you can sit back and watch your algorithms solve the maze in real-time. We have included two visualizer scripts that will boot up the Gymnasium GUI.

To see your **Value Iteration** algorithm in action, run:
```bash
python watch_value.py

```

To see your **Policy Iteration** algorithm in action, run:

```bash
python watch_policy.py

```

*If your Bellman equations are correct, you will see the agent perfectly navigate the ice and grab the treasure. If there is a bug in your math, the agent will likely fall into a hole!*

---


## How to Submit
Simply push your code to the `main` branch. 
