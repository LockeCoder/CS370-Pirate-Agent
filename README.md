# CS-370 Pirate Intelligent Agent

A Python reinforcement learning project that trains a pirate agent to navigate a maze and reach the treasure using Deep Q-Learning.

## Project Overview

This project was created for CS-370 to demonstrate reinforcement learning, neural network-based decision making, and intelligent agent development.

The goal of the project was to train an agent to solve a maze navigation problem. Instead of using hard-coded pathfinding logic, the pirate agent learns from repeated attempts, reward feedback, experience replay, and exploration/exploitation tuning.

The agent uses a Deep Q-Network to estimate the value of possible actions from each maze state. Over time, the model learns which actions are more likely to move the agent toward the treasure.

## Repository Contents

- `TreasureHuntGame.ipynb` - Main notebook containing training logic, tests, and results
- `TreasureHuntGame.html` - Exported HTML version of the notebook for quick viewing
- `TreasureMaze.py` - Maze environment used by the pirate agent
- `Project Two.docx` - Design defense and reflection document
- `.gitignore` - Git ignore configuration
- `README.md` - Repository documentation

## Features

- Trains an intelligent agent using Deep Q-Learning
- Uses a neural network to estimate action values
- Applies an epsilon-greedy strategy for exploration and exploitation
- Stores training samples in replay memory
- Uses target-network stabilization to improve training consistency
- Evaluates agent behavior in a grid-based maze environment
- Documents final training configuration and project limitations
- Includes both notebook and exported HTML formats for review

## How It Works

The maze environment represents the agent's current state, available moves, rewards, and game status. The pirate agent can choose from four possible movement actions:

- Left
- Up
- Right
- Down

The Deep Q-Network receives the maze state as input and outputs Q-values for each possible action. A higher Q-value indicates that the action is expected to produce a better long-term reward.

During training, the agent uses an epsilon-greedy policy. Early in training, the agent explores more often by choosing random actions. As training continues, epsilon decreases, and the agent increasingly chooses actions based on learned Q-values.

Replay memory stores previous experiences so the model can train from a broader sample of past states, actions, rewards, and next states instead of only learning from the most recent move. A target network is also used to stabilize Q-value updates during training.

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

## Technologies Used

- Python
- Jupyter Notebook
- Deep Q-Learning
- Neural networks
- NumPy
- Keras / TensorFlow
- Reinforcement learning environment logic

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
- Runtime-aware experimentation
- Jupyter Notebook workflow
- Technical reflection and documentation

## Technical Reflection

### Starter Code and My Contributions

The project included provided environment and experience buffer code. I kept the existing maze environment structure and expanded the training logic needed to train and evaluate the intelligent agent.

My work focused on implementing and tuning the Deep Q-Learning process, including the training loop, action-selection strategy, replay memory usage, target-network synchronization, hyperparameter tuning, and runtime-limited validation.

### Reinforcement Learning Approach

The agent learns through trial and error. Each action produces feedback through the environment's reward system, and the model uses that feedback to improve future action choices.

The main challenge was balancing exploration and exploitation. If the agent explores too much, it may fail to consistently reach the treasure. If it exploits too early, it may converge on weak behavior before learning better paths. Tuning epsilon, epsilon decay, discount factor, replay memory, and episode length helped improve training behavior.

### Challenges and Problem Solving

Sparse rewards and longer maze paths made training difficult under a limited runtime. Some starting locations are farther from the treasure and require longer successful paths before the model can receive useful reward feedback.

To address this, I adjusted the discount factor, step limits, target-network synchronization, and replay memory settings. I also prioritized the canonical top-left starting position used for grading while expanding validation across additional free starting cells when possible.

### Connection to Computer Science

This project demonstrates how computer science applies algorithmic thinking to intelligent decision-making problems. The agent does not simply follow a fixed set of instructions. Instead, the system represents a problem space, evaluates possible actions, learns from feedback, and improves through repeated training.

This connects to broader software engineering concepts such as modeling, optimization, testing, maintainability, and ethical documentation of system limitations.

### Ethics and Limitations

I documented the project limits honestly. The model's behavior depends on training time, reward structure, hyperparameter choices, and the difficulty of the starting position.

Because the final run used a project deadline time cap, the results should be interpreted within that runtime constraint. The notebook and HTML export provide the detailed training configuration and final recorded results.

## Project Value

This project shows the ability to work with machine learning concepts, reinforcement learning logic, neural network-based decision making, and experimental tuning.

It is strongest for demonstrating Python development, AI/ML fundamentals, problem solving, model evaluation, and the ability to explain technical limitations clearly.

## Future Improvements

- Add a `requirements.txt` file for easier setup
- Add clearer setup instructions for TensorFlow/Keras dependencies
- Add a short explanation of the reward structure
- Add more comparison runs with different epsilon schedules and discount factors
- Improve reporting for success rate across all valid starting cells
- Add charts showing training progress over time
- Organize supporting files into `notebooks/`, `src/`, and `docs/` folders

## Academic Portfolio Notice

This repository is shared as an academic portfolio artifact. It may include coursework documentation and assignment-specific material created for an educational setting.

Please do not reuse, submit, or redistribute this work as your own.
