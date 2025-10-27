# CS 370 – Pirate Intelligent Agent (DQN)

This repo holds my Project Two work: a pirate “intelligent agent” that learns to navigate a maze and grab the treasure using Deep Q-Learning.

- `TreasureHuntGame.ipynb` - the main notebook (training, tests, plots)
- `TreasureHuntGame.html` - exported HTML for quick viewing
- `TreasureMaze.py` - the maze environment used by the agent
- `Project Two.docx` - my design defense / reflection

## How it works (one paragraph, no fluff)

I use a **Deep Q-Network** that takes the flattened maze and outputs four Q-values (left/up/right/down). Training samples come from a replay buffer so updates are not highly correlated. A **target network** (a delayed copy of the online network) stabilizes the bootstrapped target. Actions are chosen with **ε-greedy** (start higher ε to explore; decay slowly so the agent does not lock in too early). Because rewards are sparse and some starts are far from the goal, I increased per-episode step limits and used a higher discount (γ=0.99) so successful paths influence earlier decisions.

**Key knobs I used**
- `epsilon`: start `0.35`, decay `0.999`, min `0.06`
- `gamma`: `0.99`
- `max_steps_ep`: `9 * maze.size`
- `target_sync`: `100`
- Replay: `max_memory ≈ 12 * maze.size`, batch `96`

## Results snapshot

- Single-start (top-left) test: **[True/False]**
- Completion check (all free starts): **[True/False]**
- Notes: final run used a wall-clock cap of **~25 minutes** to meet the deadline.

## What I actually did (short reflection)

- **My work vs. starter code.** I kept the environment and experience buffer, wrote the training loop logic, added the target network, handled both `'running'`/`'not_over'` status values, and tuned the ε schedule and step caps.
- **How this connects to CS.** It is the full problem-solving loop: define the objective, model the world, choose an algorithm, measure, iterate. The model is never magic, just deliberate, repeatable improvement.
- **Ethics and honesty.** I documented limits: sparse rewards and long horizons make coverage expensive. Under time limits I focused first on the canonical start used in grading, then broadened when possible. Everything needed to reproduce the result is in the notebook.

## Reproduce my final run

In the notebook’s training cell:
```python
elapsed = qtrain(
    model, maze,
    epochs=999999,
    max_memory=12*maze.size,
    data_size=96,
    gamma=0.99,
    eps_decay=0.999,
    eps_min=0.06,
    max_steps_ep=9*maze.size,
    time_cap_s=1500,      # ~25 minutes
    target_sync=100,
    verbose=25,
    start_cell=(0, 0)     # match the single-test start
)
