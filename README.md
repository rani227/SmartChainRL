# SmartChainRL
SmartChain: Reinforcement Learning for Resilient Retail Supply Chains

SmartChain is a reinforcement learning (RL) powered simulation that optimizes inventory distribution across a retail supply chain of warehouses and stores — **even under disruptions**.

Built using Python, Streamlit, and Stable-Baselines3, this project showcases how smart agents can reduce stockouts, prevent over-supply, and outperform traditional random policies in dynamic environments.

---

## Project Overview

Retail supply chains face unpredictable demand, transportation delays, and facility disruptions. Traditional systems often struggle to adapt in real time.

**SmartChain** introduces an intelligent agent using the PPO (Proximal Policy Optimization) algorithm to:

- Learn optimal shipment strategies from warehouses to stores
- Handle fluctuating demand and warehouse outages
- Minimize stockouts and avoid excess inventory
- Continuously improve through simulation-based feedback

---

## Features

-  **Reinforcement Learning Agent (PPO)**
-  **Simulation environment with disruptions**
-  **Streamlit dashboard for visualization**
-  **Comparison with baseline (random agent)**
-  **Adjustable simulation steps**
-  **KPIs: average reward, total stockouts**
-  **Shipment matrix, demand trends, warehouse stock levels**

---

## Tech Stack

- **Python 3.10+**
- **Stable-Baselines3** for PPO agent training
- **Gym** for custom environment simulation
- **Streamlit** for interactive dashboard
- **NumPy**, **Pandas**, **Matplotlib** for data handling and visualization

---

## Project Structure

smartchain/
│
├── supply_chain_discrete_env.py     # Custom OpenAI Gym environment simulating supply chain dynamics
├── train_agent.py                   # Script to train PPO agent using Stable-Baselines3
├── dashboard.py                     # Streamlit dashboard to visualize and compare agent performance
├── smartchain_ppo_default_model/    # Folder containing trained PPO model and configuration files
├── requirements.txt                 # Python dependencies required to run the project
└── README.md                        # Project overview, usage instructions, and documentation



