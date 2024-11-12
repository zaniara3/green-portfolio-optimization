## Greenhouse-Gas-Emission-Aware Portfolio Optimization with Deep Reinforcement Learning

This repository contains code and resources for Greenhouse-Gas-Emission-Aware Portfolio Optimization with Deep Reinforcement Learning, [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4999656),  a project that leverages reinforcement learning to develop an environmentally sustainable portfolio strategy. The portfolio aims to balance financial returns with greenhouse gas (GHG) emission reductions, aligning with Environmental, Social, and Governance (ESG) principles.

## Project Overview

### Objective
The objective of this project is to implement a portfolio optimization model that maximizes financial returns while minimizing risk and GHG emissions. This multi-objective optimization is achieved using reinforcement learning techniques to dynamically adjust portfolio weights across various sectors.

### Key Components
- **Reinforcement Learning (RL):** The core of this optimization leverages reinforcement learning, specifically deep policy gradient-type reinforcement learning (RL) algorithm, to adjust sector weights dynamically.
- **Mean-Variance Optimization:** The model incorporates a mean-variance objective alongside GHG considerations, aiming to optimize for both risk-adjusted returns and environmental sustainability.
- **GHG Emission Data:** Real or simulated GHG emission data is used to assess and guide the environmental impact of the portfolio's allocations.

## Features

- **Dynamic Portfolio Rebalancing:** Continuous portfolio adjustment based on market conditions and emission objectives.
- **Risk Management:** Incorporates variance to mitigate the risk.

## Data
- **Simulated Returns:** Including all returns, volatility and dynamic covariance matrice of sectors, that are simulated based on [DCC-EGARCH](https://github.com/zaniara3/DCC-EGARCH-Simulation) presented by the same authors. 
- **Simulated GHG:** Simulated GHG intensities of sectors 
"# green-portfolio-optimization" 
