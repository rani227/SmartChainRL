from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from supply_chain_discrete_env import SupplyChainDiscreteEnv


env = DummyVecEnv([lambda: SupplyChainDiscreteEnv()])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    learning_rate=3e-4,
    policy_kwargs={"net_arch": [128, 128]},
)

model.learn(total_timesteps=100_000)
model.save("smartchain_ppo_default_model")
