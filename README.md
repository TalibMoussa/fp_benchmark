# LIF Neuron Simulator with Fokker‑Planck Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Proposal](https://img.shields.io/badge/Research-Proposal-blue)](moussa_fp_research.pdf)

**Author:** Abdulrahman Moussa  
**Affiliation:** Department of Biochemistry
**Contact:** talibmoussa23@gmail.com  

This repository contains a **computational framework** for simulating Leaky Integrate‑and‑Fire (LIF) neurons, both deterministic and stochastic, and for benchmarking against the analytical Fokker‑Planck stationary solution of the equivalent Ornstein‑Uhlenbeck process. It is the first concrete implementation of the research proposal:

> *“Information‑Geometric Optimization of Fokker‑Planck Dynamics in Neuromorphic Synthetic Synapses: A Framework for Near‑Equilibrium Distribution Learning”*

## Key Features

- **Deterministic LIF** – Euler integration with refractory period.
- **Stochastic LIF** – Euler‑Maruyama integration with Gaussian white noise (ensemble support).
- **Fokker‑Planck analysis** – analytical stationary PDF (OU) + finite‑volume numerical time evolution.
- **Benchmark metrics** – Mean Squared Error (MSE) and Kullback‑Leibler divergence between numerical Fokker‑Planck and Monte Carlo histograms as a function of noise amplitude σ.
- **Publication‑ready figure** – 5‑panel summary: (A) deterministic traces, (B) stochastic example trace, (C) ensemble histogram, (D) FP stationary PDF (numerical vs analytical), (E) MSE and KL divergence vs σ.

## Requirements

- Python 3.8+
- `numpy`
- `scipy`
- `matplotlib`

Install with:
```bash
pip install numpy scipy matplotlib
