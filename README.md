# THERMOW

**THERMOW** (THreshold, gREedy, MDP, Optimization, and Window-based strategies) is an interactive simulation tool for evaluating and comparing energy optimization algorithms in data centers.

Developed as part of a doctoral research project, THERMOW provides a GUI-based interface to simulate various scheduling strategies aimed at reducing energy consumption while maintaining performance.

## ✨ Features

- Simulation of real-time workload arrival
- Implementation of:
  - Threshold-based algorithms
  - Greedy and Greedy Window heuristics
  - Full and Truncated Markov Decision Processes (MDP)
- Generation of LaTeX-ready result reports and plots
- Optional PRISM model export for formal verification

## 📦 Requirements

Install the dependencies via pip:

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

Run the graphical interface:

```bash
python globle.15.py
```

Then:

1. Load your job arrival histogram (`CSV` format)
2. Set simulation parameters (buffer size, cost values, etc.)
3. Select optimization methods
4. Launch the simulation and visualize results

## 📁 Output

- Graphs of cost, time, and memory for each method
- Optional PRISM model file
- LaTeX report including figures and analysis

## 📄 License

This project is licensed under the **MIT License**.

## 👨‍🔬 Author

Developed during the PhD research of Léa BAYATI  
Feel free to contribute or reach out for collaborations in energy-aware computing.
