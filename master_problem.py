from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
from generate_instances import ReplenishmentConfig, generate_instance
import pandas as pd
from tabulate import tabulate

def solve_rmp(instance):
    cfg = instance["config"]
    products = range(cfg.num_products)
    stores = range(cfg.num_stores)
    time_horizon = cfg.time_horizon
    
    model = LpProblem("Retail_Replenishment_RMP", LpMinimize)
    
    # Decision Variables
    x = LpVariable.dicts("Order", (stores, products, range(time_horizon)), 
                        lowBound=0, cat="Integer")
    y = LpVariable.dicts("OrderFlag", (stores, products, range(time_horizon)), 
                        cat="Binary")
    I = LpVariable.dicts("Inventory", (stores, products, range(time_horizon)), 
                        lowBound=0)
    u = LpVariable.dicts("Shortage", (stores, products, range(time_horizon)), 
                        lowBound=0)
    
    # Objective Function
    model += lpSum(
        cfg.ordering_cost[p] * y[s][p][t] +
        cfg.holding_cost[p] * I[s][p][t] +
        cfg.shortage_cost[p] * u[s][p][t]
        for s in stores for p in products for t in range(time_horizon)
    )
    
    # Constraints
    for s in stores:
        for p in products:
            for t in range(time_horizon):
                # Inventory balance
                if t == 0:
                    prev_inv = instance["initial_inventory"][(s, p)]
                else:
                    prev_inv = I[s][p][t-1]
                
                model += (
                    I[s][p][t] == prev_inv + 
                    x[s][p][t] * cfg.case_pack_size[p] - 
                    (instance["demand"][(s, p, t)] - u[s][p][t]),
                    f"InvBalance_{s}_{p}_{t}"
                )
                
                # Order activation constraints
                model += (
                    x[s][p][t] >= y[s][p][t] * cfg.min_order_qty[p],
                    f"MOQ_Lower_{s}_{p}_{t}"
                )
                model += (
                    x[s][p][t] <= y[s][p][t] * 1000,  # Big-M value
                    f"MOQ_Upper_{s}_{p}_{t}"
                )
    
    # Warehouse capacity constraints
    for p in products:
        model += (
            lpSum(x[s][p][t] * cfg.case_pack_size[p] 
                 for s in stores for t in range(time_horizon)) <= cfg.warehouse_capacity[p],
            f"WarehouseCap_{p}"
        )
    
    # Solve
    model.solve()
    
    # Build time-series table
    results = []
    for s in range(cfg.num_stores):
        for p in range(cfg.num_products):
            for t in range(cfg.time_horizon):
                results.append({
                    "Store": f"Location_{s}",
                    "Product": f"SKU_{p}",
                    "Day": t,
                    "Start_Inventory": instance["initial_inventory"][(s, p)] if t == 0 
                                      else I[s][p][t-1].varValue,
                    "Order_Placed": y[s][p][t].varValue,
                    "Case_Packs_Ordered": x[s][p][t].varValue,
                    "Units_Ordered": x[s][p][t].varValue * cfg.case_pack_size[p],
                    "Demand": instance["demand"][(s, p, t)],
                    "Shortage": u[s][p][t].varValue,
                    "End_Inventory": I[s][p][t].varValue
                })
    
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    # Example test instance
    cfg = ReplenishmentConfig(
        num_products=3,
        num_stores=2,
        time_horizon=3
    )
    instance = generate_instance(cfg)
    results_df = solve_rmp(instance)
    print("\nTime-Series Replenishment Plan:")
    print(tabulate(results_df, headers='keys', tablefmt='psql', showindex=False))
