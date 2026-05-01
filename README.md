# CS-370 Pirate Intelligent Agent

This repository contains my CS-370 Project Two work: a pirate intelligent agent that learns to navigate a maze and reach the treasure using Deep Q-Learning.

The project demonstrates reinforcement learning concepts, including state representation, action selection, reward-based learning, experience replay, target-network stabilization, hyperparameter tuning, and model evaluation in a grid-based environment.

## Project Overview

The goal of this project was to train an intelligent agent to solve a maze navigation problem. The pirate agent learns how to move through the maze and reach the treasure by using a Deep Q-Network to estimate the value of possible actions from each game state.

The agent uses reinforcement learning rather than hard-coded pathfinding. It improves through repeated training episodes, reward feedback, replay memory, and gradual exploration/exploitation tuning.

## Repository Contents

- `TreasureHuntGame.ipynb` — main notebook containing training logic, tests, and results
- `TreasureHuntGame.html` — exported HTML version of the notebook for quick viewing
- `TreasureMaze.py` — maze environment used by the pirate agent
- `Project Two.docx` — design defense and reflection document
- `README.md` — repository documentation
- `.gitignore` — Git ignore configuration

## How It Works

The project uses a Deep Q-Network that takes the flattened maze state as input and outputs Q-values for four possible actions:

- Left
- Up
- Right
- Down

Training samples are stored in a replay buffer so updates are not overly dependent on the most recent sequence of actions. A target network, which is a delayed copy of the online network, helps stabilize bootstrapped Q-value targets during training.

Actions are selected using an epsilon-greedy strategy. Early in training, the agent explores more often. Over time, epsilon decays so the agent increasingly selects actions based on learned Q-values.

Because rewards are sparse and some starting locations are far from the goal, the training configuration uses a higher discount factor and increased per-episode step limits so successful paths can influence earlier decisions.

## Key Training Parameters

- `epsilon`: starts at `0.35`
- `eps_decay`: `0.999`
- `eps_min`: `0.06`
- `gamma`: `0.99`
- `max_steps_ep`: `9 * maze.size`
- `target_sync`: `100`
- Replay memory: approximately `12 * maze.size`
- Batch size: `96`
- Time cap: approximately 25 minutes

## Results Snapshot

The final training and validation results are documented in `TreasureHuntGame.ipynb` and the exported `TreasureHuntGame.html` file.

The final run used a wall-clock cap of approximately 25 minutes to meet the project deadline. Under that constraint, the training configuration prioritized the canonical top-left starting position used for grading, then expanded validation across additional free starting cells when possible.

## Reproduce the Final Run

In the notebook training cell, use:

```python
elapsed = qtrain(
    model, maze,
    epochs=999999,
    max_memory=12 * maze.size,
    data_size=96,
    gamma=0.99,
    eps_decay=0.999,
    eps_min=0.06,
    max_steps_ep=9 * maze.size,
    time_cap_s=1500,
    target_sync=100,
    verbose=25,
    start_cell=(0, 0)
)
```

## Skills Demonstrated

- Python programming
- Reinforcement learning fundamentals
- Deep Q-Learning
- Neural network-based decision making
- Experience replay
- Target-network stabilization
- Epsilon-greedy exploration
- Hyperparameter tuning
- Model evaluation
- Jupyter Notebook workflows
- Technical reflection and documentation

## Technologies Used

- Python
- Jupyter Notebook
- Deep Q-Learning
- Neural networks
- NumPy
- Keras / TensorFlow
- Reinforcement learning environment logic

## Reflection

### My Work vs. Starter Code

I kept the provided environment and experience buffer structure, then added and adjusted the training loop logic needed to train the agent. I added the target network, handled both `running` and `not_over` game status values, and tuned the epsilon schedule, discount factor, replay settings, target synchronization interval, and episode step limits.

### Connection to Computer Science

This project represents the full computer science problem-solving process: define the objective, model the environment, choose an algorithm, evaluate results, and iterate. The model is not magic. It improves because the problem is represented clearly, feedback is measured consistently, and the algorithm is adjusted based on observed behavior.

### Ethics and Limitations

I documented the project limits honestly. Sparse rewards and long maze paths make full coverage expensive, especially under runtime constraints. Under the project deadline, I prioritized the canonical starting position used for grading and then expanded validation when possible. The notebook includes the training configuration needed to reproduce the final run.

## Future Improvements

- Add a `requirements.txt` file for easier setup
- Add a short explanation of the reward structure
- Add more comparison runs with different epsilon schedules and discount factors
- Improve reporting for success rate across all valid starting cells
- Organize supporting files into `notebooks/`, `src/`, and `docs/` folders
