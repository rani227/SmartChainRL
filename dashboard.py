import streamlit as st
import numpy as np
import pandas as pd
from supply_chain_discrete_env import SupplyChainDiscreteEnv
from stable_baselines3 import PPO

st.set_page_config(layout="wide")
st.title("📦 SmartChain RL Agent Simulation Dashboard")


model = PPO.load("smartchain_ppo_default_model")

st.sidebar.header("⚙️ Simulation Settings")
num_steps = st.sidebar.slider("Number of steps", min_value=20, max_value=200, value=100)
compare_with_random = st.sidebar.checkbox("Compare with Random Agent", value=True)

def run_simulation(agent=None):
    env = SupplyChainDiscreteEnv()
    obs, _ = env.reset()

    warehouse_stock, store_demand, reward_list, shipment_log, stockouts = [], [], [], [], []

    for _ in range(num_steps):
        if agent == "ppo":
            action, _ = model.predict(obs, deterministic=True)
        elif agent == "random":
            action = env.action_space.sample()
        else:
            raise ValueError("Unsupported agent")

        obs, reward, terminated, truncated, _ = env.step(action)
        warehouse_stock.append(env.warehouse_stock.copy())
        store_demand.append(env.store_demand.copy())
        reward_list.append(reward)
        shipment_log.append(action)
        stockouts.append(np.sum(np.maximum(env.store_demand - np.sum(action, axis=0), 0)))

    return {
        "warehouse": warehouse_stock,
        "demand": store_demand,
        "reward": reward_list,
        "shipment": shipment_log,
        "stockout": stockouts
    }

st.subheader("🤖 PPO Agent Performance")
ppo_data = run_simulation(agent="ppo")

st.subheader("📊 Warehouse Stock Levels Over Time")
df_warehouse = pd.DataFrame(ppo_data["warehouse"], columns=[f"W{i+1}" for i in range(3)])
st.line_chart(df_warehouse)

st.subheader("🛒 Store Demands Over Time")
df_demand = pd.DataFrame(ppo_data["demand"], columns=[f"S{i+1}" for i in range(5)])
st.line_chart(df_demand)

st.subheader("💰 Reward Over Time")
df_reward = pd.DataFrame({"Reward": ppo_data["reward"]})
st.line_chart(df_reward)

avg_reward = np.mean(ppo_data["reward"])
total_stockouts = np.sum(ppo_data["stockout"])
st.success(f"✅ **Average Reward:** {avg_reward:.2f}")
st.error(f"❌ **Total Stockouts:** {total_stockouts}")

if compare_with_random:
    st.subheader("🎲 Random Agent Comparison")
    random_data = run_simulation(agent="random")
    st.line_chart(pd.DataFrame({"PPO Reward": ppo_data["reward"], "Random Reward": random_data["reward"]}))
    st.warning(f"📉 Random Agent Avg Reward: {np.mean(random_data['reward']):.2f}")
