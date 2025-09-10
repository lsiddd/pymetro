# PyMetro

A Python implementation of Mini Metro because apparently the original game wasn't complicated enough.

## What Is This

PyMetro is a recreation of the subway management game Mini Metro, built with pygame. It also includes reinforcement learning agents that can play the game, presumably better than you can.

The game involves drawing subway lines between stations to transport passengers. Passengers have shapes, stations have shapes, and passengers want to go to stations with matching shapes. This is about as complex as public transportation gets.

## Features

- **Playable Game**: Click and drag to build subway lines
- **Multiple Cities**: London, Paris, New York, and Tokyo (because every metro system is exactly the same)
- **AI Agents**: Deep Q-Network agents that have learned to optimize subway systems
- **Real-time Simulation**: Watch your transit empire crumble in real time
- **Resource Management**: Limited trains, carriages, and interchanges (just like real life)

## Installation

```bash
pip install -r requirements.txt
```

That's it. Modern dependency management is a beautiful thing.

## Usage

### Play the Game

```bash
python main.py
```

Controls:
- Left click: Add stations to lines
- Right click: Remove stations from lines
- Space: Pause/unpause (for when reality becomes too much)
- 1-9: Select line colors
- ESC: Exit (the coward's way out)

### Train an AI Agent

```bash
python train.py
```

Watch as artificial intelligence slowly learns what you could have figured out in minutes.

### Evaluate a Trained Agent

```bash
python evaluate.py
```

See how the AI performs. Spoiler: it's probably better than your attempts.

## Project Structure

```
pymetro/
├── components/          # Game objects (stations, trains, passengers)
├── systems/            # Game systems (rendering, input, pathfinding)
├── rl_agent/           # Reinforcement learning components
├── main.py             # Game entry point
├── train.py            # AI training script
├── evaluate.py         # AI evaluation script
└── config.py           # Game configuration
```

## How It Works

The game simulates a growing transit system where:

1. Stations spawn randomly with different shapes
2. Passengers spawn at stations wanting to go to matching shapes
3. You draw lines connecting stations
4. Trains run on these lines carrying passengers
5. Everything eventually goes wrong

The RL agent uses a Deep Q-Network to learn optimal line placement and resource allocation. It observes the game state and takes actions to maximize passenger throughput while minimizing overcrowding.

## Dependencies

- pygame: For rendering and input
- torch: For the neural networks
- numpy: For the math
- matplotlib: For pretty graphs
- tensorboard: For prettier graphs

## Contributing

Pull requests welcome. Bug reports also welcome, though they're probably features.

## License

This project exists in the public domain, much like abandoned subway stations.

---

*"The best transit system is the one that doesn't exist." - Every commuter, probably*