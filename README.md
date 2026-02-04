# **Bons-AI: An Agent-Based Model to Evaluate the Behavior of Bonsai Growers According to Different Levels of Communication and Experience**  
*AAMAS 2026 - Cyprus*


Knowledge transfer and social learning are fundamental challenges in multi-agent systems (MAS), particularly in domains where decisions require long term knowledge and environmental factors. In this paper, we introduce Bons-AI, an agent-based model (ABM) that simulates the interaction between bonsai growers with different levels of expertise, aiming to investigate how experience and communication affect the health and style preservation of bonsais. Our model integrates Q-learning with climatic and biological conditions, to simulate plant growth and human decisions. We conducted experiments comparing scenarios with inexperienced growers, autonomous learners, and master–apprentice relationships. The results show that knowledge sharing reduces mortality by 18\% and increases overall health by 9.5\%, highlighting the role of social communication in the learning process. Beyond the specific domain of bonsai cultivation, this work contributes to the MAS by offering a framework for studying adaptive behavior, distinct expertise level, and communication based knowledge transfer in complex environments.

---

## Features

- Simulates **multiple bonsai trees** with 5 classic styles:
  - Formal Upright
  - Informal Upright
  - Slanting
  - Cascade
  - Semi-Cascade
- AI Growers perform realistic actions:
  - Watering
  - Pruning
  - Wiring / Unwiring
  - Fertilizing
  - Repotting
- Growth modeled with a **generalized logistic growth function**
- Seasonal rainfall and environmental noise based on real climate data
- Experiments with:
  - Inexperienced growers
  - Learning growers (Q-learning)
  - Expert growers
  - Master–apprentice communication
- Automatic data collection and visualization


## Project Structure

```text
Bons-AI/
├── agents/
│   ├── bonsai_agent.py        # Bonsai biological model
│   └── caregiver_agent.py    # Grower agent with RL
├── model/
│   └── bonsai_model.py       # Mesa model and scheduler
├── config/
│   └── parameters.py         # Global simulation parameters
├── utils/
│   ├── plotting.py           # Visualization utilities
│   └── statistics.py         # Metrics and analysis
├── main.py                   # Entry point for experiments
├── Results/                  # Auto-generated outputs
└── README.md                 # You are here
```

---

## Requirements

- Python **3.10+**
- Dependencies:

```bash
pip install mesa pandas matplotlib numpy tabulate
```

---

## How to Run

Run the full experimental pipeline with:

```bash
python3 main.py
```

This will:
1. Execute multiple simulation runs (default: 10)
2. Simulate up to **10 years** of bonsai growth
3. Save results to the `Results/` directory
4. Generate plots and tables

---

## Configuration

Simulation parameters are defined in:

```text
config/parameters.py
```

You can adjust:
- Simulation duration (years, steps)
- Environmental decay rates
- Growth thresholds
- Caregiver action limits
- Learning parameters (α, γ, ε)

---

## Authors

Sara Satake · Guilherme Nakahata · Claus Aranha  
University of Tsukuba