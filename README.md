# Lunar Lander DQN

A Deep Q-Network (DQN) agent trained to solve Gymnasium's `LunarLander-v3` environment — landing a spacecraft safely between two flags by controlling a main engine and two lateral thrusters.

The best checkpoint solves the environment (average reward ≥ 250 over 100 consecutive episodes) in 1797 episodes.

## Environment

- **State (8-dim):** x/y position, x/y velocity, angle, angular velocity, left/right leg contact
- **Actions (4, discrete):** do nothing, fire left engine, fire main engine, fire right engine
- **Reward:** shaped for proximity to the pad, leg contact, and safe landing; penalized for crashing and for engine fuel use

## Architecture

- 2-hidden-layer MLP (64 → 64, ReLU), mapping the 8-dim state to 4 Q-values
- Experience replay buffer (size 100,000) with uniform random sampling
- Separate local/target networks with soft (Polyak) target updates
- Epsilon-greedy exploration with exponential decay

The agent/replay-buffer structure follows the standard DQN reference pattern popularized by Udacity's Deep Reinforcement Learning Nanodegree materials (local/target network pair + soft update, fixed-size replay buffer) rather than a from-scratch design — noted here for transparency.

## Results

Three hyperparameter configurations were compared:

| Config | Learning rate | ε-decay | Result |
|---|---|---|---|
| **Baseline** | 5e-4 | 0.997 | Solved in 1797 episodes, final avg score 250.40 |
| **Fast Learner** | 1e-3 | 0.990 | Solved in 1389 episodes, final avg score 250.10 (higher training variance) |
| **Precision Pilot** | 1e-4 | 0.999 | Did not solve within 2000 episodes (avg score 92.98) |

Full analysis — including training curves, exploration/target-network tradeoffs, and observed failure modes (reward hacking via "swooping" descents, catastrophic collapse in the slow-decay run) — is in [`report/Report_Monish_Balu.pdf`](report/Report_Monish_Balu.pdf).

An additional exploratory run (`checkpoints/experiments/checkpoint_accurate.pth`) was trained but isn't covered in the report.

## Repo structure

```
src/
  model.py          Q-network (MLP)
  dqn_agent.py       Agent, replay buffer, learning step
  train.py           Training loop (produces checkpoints/checkpoint_best.pth)
  watch_agent.py      Render a trained agent live (render_mode="human")
  record_video.py    Record MP4s of a trained agent
checkpoints/
  checkpoint_best.pth              Baseline config, solved at episode 1797
  experiments/                     Other hyperparameter runs from the report
videos/                            Recorded evaluation episodes
report/                            Written analysis
```

## Running it

```bash
pip install -r requirements.txt

cd src
python train.py            # trains from scratch, saves to ../checkpoints/checkpoint_best.pth
python watch_agent.py      # watch the trained agent (requires a display)
python record_video.py     # record evaluation episodes to ../videos/
```

## License

MIT — see [LICENSE](LICENSE).
