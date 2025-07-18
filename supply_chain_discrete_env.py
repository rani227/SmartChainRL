import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SupplyChainDiscreteEnv(gym.Env):
    def __init__(self):
        super(SupplyChainDiscreteEnv, self).__init__()
        self.num_warehouses = 3
        self.num_stores = 5
        self.bucket_levels = 11
        self.max_stock = 150
        self.max_demand = 30

        self.action_space = spaces.MultiDiscrete([self.bucket_levels] * (self.num_warehouses * self.num_stores))
        self.observation_space = spaces.Box(low=0, high=200, shape=(self.num_warehouses + self.num_stores,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.warehouse_stock = np.full(self.num_warehouses, self.max_stock)
        self.store_demand = np.random.randint(10, self.max_demand, size=self.num_stores)
        obs = np.concatenate((self.warehouse_stock, self.store_demand)).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.current_step += 1
        fulfilled = np.zeros(self.num_stores)
        delivery_cost = 0
        buffer_bonus = 0
        idx = 0

        allocations = np.zeros((self.num_warehouses, self.num_stores))
        for w in range(self.num_warehouses):
            for s in range(self.num_stores):
                qty_requested = action[idx] * 10
                qty_to_send = min(qty_requested, self.warehouse_stock[w])
                allocations[w][s] = qty_to_send
                fulfilled[s] += qty_to_send
                self.warehouse_stock[w] -= qty_to_send
                delivery_cost += 0.4 * qty_to_send
                idx += 1

        stockout_penalty = np.sum(np.maximum(0, self.store_demand - fulfilled)) * 1.5
        oversupply_penalty = np.sum(np.maximum(0, fulfilled - self.store_demand)) * 0.3

        if np.random.rand() < 0.05:
            disrupted = np.random.randint(0, self.num_warehouses)
            self.warehouse_stock[disrupted] = 0

        avg_demand = np.mean(self.store_demand)
        replenishment = np.random.randint(int(avg_demand / 2), int(avg_demand * 1.2), size=self.num_warehouses)
        self.warehouse_stock = np.minimum(self.warehouse_stock + replenishment, self.max_stock)

        for stock in self.warehouse_stock:
            if stock >= 0.3 * self.max_stock:
                buffer_bonus += 5

        self.store_demand = np.random.randint(10, self.max_demand, size=self.num_stores)
        reward = - (delivery_cost + stockout_penalty + oversupply_penalty) + buffer_bonus


        obs = np.concatenate((self.warehouse_stock, self.store_demand)).astype(np.float32)
        return obs, reward, False, False, {}
