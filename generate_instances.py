import random
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class ReplenishmentConfig:
    # Problem dimensions
    num_products: int = 10       # Number of SKUs
    num_stores: int = 5          # Number of retail locations
    time_horizon: int = 7        # Days in planning period
    
    # Capacity parameters
    warehouse_capacity: Dict[int, int] = None  # ProductID -> Max units
    
    # Cost parameters
    ordering_cost: Dict[int, float] = None     # ProductID -> Cost/order
    holding_cost: Dict[int, float] = None      # ProductID -> Cost/unit/day
    shortage_cost: Dict[int, float] = None     # ProductID -> Cost/unit shortage
    
    # Demand parameters
    base_demand: Dict[int, float] = None       # ProductID -> Avg daily demand
    demand_variance: float = 0.3               # ±30% daily variation
    
    # Supplier constraints
    min_order_qty: Dict[int, int] = None       # ProductID -> MOQ
    case_pack_size: Dict[int, int] = None      # ProductID -> Units/case

def generate_instance(cfg: ReplenishmentConfig):
    """Generates a random problem instance with configurable parameters"""
    random.seed(42)
    products = list(range(cfg.num_products))
    stores = list(range(cfg.num_stores))
    
    # Generate default values if not provided
    if cfg.warehouse_capacity is None:
        cfg.warehouse_capacity = {
            p: max(50, int((cfg.num_stores * cfg.num_products * cfg.time_horizon * random.uniform(0.5, 1.5))))
            for p in products
        }

        
    if cfg.ordering_cost is None:
        cfg.ordering_cost = {p: round(random.uniform(5, 20), 2) for p in products}
        
    if cfg.holding_cost is None:
        cfg.holding_cost = {p: round(random.uniform(0.1, 1.5), 2) for p in products}
        
    if cfg.shortage_cost is None:
        cfg.shortage_cost = {p: round(random.uniform(3, 10), 2) for p in products}
        
    if cfg.base_demand is None:
        cfg.base_demand = {p: random.randint(5, 30) for p in products}
        
    if cfg.min_order_qty is None:
        cfg.min_order_qty = {p: random.choice([1, 5, 10]) for p in products}
        
    if cfg.case_pack_size is None:
        cfg.case_pack_size = {p: random.randint(5, 25) for p in products}

    total_capacity = {p: cfg.warehouse_capacity[p] * cfg.time_horizon for p in products}

    # Generate stochastic demand
    demand = {}
    for p in products:
        for t in range(cfg.time_horizon):
            for s in stores:
                max_demand = total_capacity[p] / (cfg.num_stores * cfg.time_horizon)
                demand[(s, p, t)] = max(0, int(
                    np.random.poisson(max_demand * random.uniform(0.5, 1.5))
                ))
    
    return {
        "config": cfg,
        "demand": demand,
        "initial_inventory": { (s, p): random.randint(0, 20) 
                              for s in stores for p in products }
    }

# Example usage:
if __name__ == "__main__":
    cfg = ReplenishmentConfig(
        num_products=3,
        num_stores=2,
        time_horizon=3
    )
    
    instance = generate_instance(cfg)
    print("Generated instance with:")
    print(f"- {cfg.num_products} products across {cfg.num_stores} stores")
    print(f"- Demand: {list(instance['demand'].items())}")
    print(f"- Ordering costs: {cfg.ordering_cost}")
    print(f"- Initial inventory: {instance['initial_inventory']}")
    print(f"- Warehouse capacity: {cfg.warehouse_capacity}")
    print(f"- Min order qty: {cfg.min_order_qty}")
    print(f"- Case pack size: {cfg.case_pack_size}")
    print(f"- Time horizon: {cfg.time_horizon} days")
    print(f"- Demand variance: {cfg.demand_variance:.0%}")

# demand[(store_idx, product_idx, day)] = units_needed
# initial_inventory[(store_idx, product_idx)] = stock_on_hand
# warehouse_capacity[product_idx] = max_units
# ordering_cost[product_idx] = cost/order
# holding_cost[product_idx] = cost/unit/day
# shortage_cost[product_idx] = cost/unit shortage
# min_order_qty[product_idx] = MOQ
# case_pack_size[product_idx] = units/case
# time_horizon = days

