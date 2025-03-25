from pulp import LpProblem, LpMinimize, lpSum, LpVariable, LpStatus, value
import pandas as pd
from generate_instances import generate_instance, ReplenishmentConfig
from math import ceil
import math

class ColumnGenerationSolver:
    def __init__(self, instance):
        self.instance = instance
        self.cfg = instance["config"]
        self.columns = self._initialize_columns()

    def _initialize_columns(self):
        """Generate columns that attempt to fulfill demand for all days"""
        columns = []
        for s in range(self.cfg.num_stores):
            for p in range(self.cfg.num_products):
                # Column 1: Aggressive ordering to meet all demand
                orders = [0] * self.cfg.time_horizon
                shortage = [0] * self.cfg.time_horizon
                inv = self.instance["initial_inventory"][(s, p)]
                
                for t in range(self.cfg.time_horizon):
                    current_demand = self.instance["demand"][(s, p, t)]
                    
                    # Calculate needed units considering current inventory
                    needed = max(0, current_demand - inv)
                    if needed > 0:
                        # Calculate minimum packs needed (respecting MOQ)
                        packs = max(
                            math.ceil(needed / self.cfg.case_pack_size[p]),
                            self.cfg.min_order_qty[p]
                        )
                        orders[t] = packs
                        inv += packs * self.cfg.case_pack_size[p]
                    
                    # Fulfill demand and calculate carryover/shortage
                    inv -= current_demand
                    if inv < 0:
                        shortage[t] = abs(inv)
                        inv = 0
                
                # Calculate total cost for this strategy
                total_cost = sum(
                    self.cfg.ordering_cost[p] * (1 if orders[t] > 0 else 0) +
                    self.cfg.holding_cost[p] * inv +
                    self.cfg.shortage_cost[p] * shortage[t]
                    for t in range(self.cfg.time_horizon)
                )
                
                columns.append({
                    'store': s,
                    'product': p,
                    'orders': orders,
                    'shortage': shortage,
                    'cost': total_cost
                })
                
                # Column 2: No orders (worst case scenario)
                columns.append({
                    'store': s,
                    'product': p,
                    'orders': [0] * self.cfg.time_horizon,
                    'shortage': [self.instance["demand"][(s, p, t)] 
                                for t in range(self.cfg.time_horizon)],
                    'cost': sum(
                        self.cfg.shortage_cost[p] * self.instance["demand"][(s, p, t)]
                        for t in range(self.cfg.time_horizon)
                    )
                })
        
        return columns
                
    def solve_rmp(self):
        """Solves Restricted Master Problem with current columns"""
        rmp = LpProblem("RMP", LpMinimize)
        
        # Lambda variables for column selection
        lambda_vars = LpVariable.dicts("lambda", range(len(self.columns)), 
                                      lowBound=0, cat="Binary")
        
        # Objective: Minimize total cost of selected columns
        rmp += lpSum(col['cost'] * lambda_vars[i] 
                    for i, col in enumerate(self.columns))
        
        # Demand covering constraints
        for s in range(self.cfg.num_stores):
            for p in range(self.cfg.num_products):
                for t in range(self.cfg.time_horizon):
                    rmp += lpSum(col['orders'][t] * lambda_vars[i] 
                                for i, col in enumerate(self.columns) 
                                if col['store'] == s and col['product'] == p) \
                            >= self.instance["demand"][(s,p,t)], \
                            f"Demand_{s}_{p}_{t}"
        
        rmp.solve()
        
        # Get dual values for pricing
        duals = {c: rmp.constraints[c].pi for c in rmp.constraints}
        
        return rmp, value(rmp.objective), duals
    
    def pricing_problem(self, duals):
        """Finds new columns with negative reduced cost"""
        new_columns = []
        
        for s in range(self.cfg.num_stores):
            for p in range(self.cfg.num_products):
                # Create subproblem for each (store, product)
                sp = LpProblem("Pricing", LpMinimize)
                
                # Subproblem variables
                x = LpVariable.dicts("x", range(self.cfg.time_horizon), 
                                    lowBound=0, cat="Integer")
                y = LpVariable.dicts("y", range(self.cfg.time_horizon), 
                                    cat="Binary")
                
                # Reduced cost calculation (FIXED TIME INDEXING)
                sp += lpSum(
                    self.cfg.ordering_cost[p] * y[t] +
                    self.cfg.holding_cost[p] * x[t] * self.cfg.case_pack_size[p]
                    for t in range(self.cfg.time_horizon)
                ) - lpSum(
                    duals.get(f"Demand_{s}_{p}_{t}", 0) * x[t] 
                    for t in range(self.cfg.time_horizon)
                )
                
                # Subproblem constraints (ADDED TIME LOOP)
                for t in range(self.cfg.time_horizon):
                    sp += x[t] <= 100 * y[t], f"BigM_{t}"
                    sp += x[t] >= self.cfg.min_order_qty[p] * y[t], f"MOQ_{t}"
                    
                sp.solve()
                
                if value(sp.objective) < -1e-5:
                    new_col = {
                        'store': s,
                        'product': p,
                        'orders': [x[t].varValue for t in range(self.cfg.time_horizon)],
                        'cost': sum(
                            self.cfg.ordering_cost[p] * y[t].varValue +
                            self.cfg.holding_cost[p] * x[t].varValue * self.cfg.case_pack_size[p]
                            for t in range(self.cfg.time_horizon)
                        )
                    }
                    new_columns.append(new_col)
        
        return new_columns

    
    def solve(self, optimality_gap_threshold=1e-5, max_iter=1000):
        """Column Generation loop"""
        best_lower_bound = float('inf')
        i = 0

        while i < max_iter:
            rmp, obj, duals = self.solve_rmp()
            new_cols = self.pricing_problem(duals)
            
            if not new_cols:
                print(f"Optimal after {i+1} iterations")
                break
                
            self.columns.extend(new_cols)
            current_obj_value = obj
            best_lower_bound = min(best_lower_bound, current_obj_value)
            optimality_gap = (current_obj_value - best_lower_bound) / current_obj_value
            print(f"Iteration {i+1}: Optimality gap: {100 * optimality_gap:.2f}%")
            
            if optimality_gap < optimality_gap_threshold:
                print(f"Optimality gap threshold reached after {i+1} iterations")
                break

            i += 1

        # Final solution
        final_schedule = self.generate_schedule()
        final_obj_value = value(rmp.objective)
        final_optimality_gap = (final_obj_value - best_lower_bound) / final_obj_value
        print(f"Final Optimality Gap: {100 * final_optimality_gap:.2f}%")
        return final_schedule

    def generate_schedule(self):
        """Convert lambda variables to time-series schedule"""
        # Solve final RMP with integer constraints
        final_rmp = LpProblem("FinalRMP", LpMinimize)
        lambda_vars = LpVariable.dicts("lambda", range(len(self.columns)), 
                                    lowBound=0, cat="Integer")
        
        final_rmp += lpSum(col['cost'] * lambda_vars[i] 
                        for i, col in enumerate(self.columns))
        
        # Add demand constraints
        for s in range(self.cfg.num_stores):
            for p in range(self.cfg.num_products):
                for t in range(self.cfg.time_horizon):
                    final_rmp += lpSum(col['orders'][t] * lambda_vars[i] 
                                    for i, col in enumerate(self.columns) 
                                    if col['store'] == s and col['product'] == p) \
                                >= self.instance["demand"][(s,p,t)]
        
        final_rmp.solve()
        
        # Check solution status first
        if final_rmp.status != 1:
            print("No feasible solution found")
        
        # Build schedule with safe value checking and days inventory left
        schedule = []
        inventory = {(s, p): self.instance["initial_inventory"][(s, p)] 
                    for s in range(self.cfg.num_stores) 
                    for p in range(self.cfg.num_products)}

        for i in range(len(self.columns)):
            var_value = value(lambda_vars[i])  # Safe value extraction
            if var_value and var_value > 0:
                col = self.columns[i]
                for t in range(self.cfg.time_horizon):
                    if col['orders'][t] > 0:
                        s, p = col['store'], col['product']
                        units_ordered = col['orders'][t] * self.cfg.case_pack_size[p]
                        inventory[(s, p)] += units_ordered
                        demand = self.instance["demand"][(s, p, t)]
                        inventory[(s, p)] = max(0, inventory[(s, p)] - demand)
                        
                        # Calculate days inventory left
                        future_demand = sum(self.instance["demand"][(s, p, t2)] 
                                            for t2 in range(t+1, self.cfg.time_horizon))
                        if future_demand > 0:
                            days_inventory_left = min(
                                self.cfg.time_horizon - t - 1,
                                inventory[(s, p)] / (future_demand / (self.cfg.time_horizon - t - 1))
                            )
                        else:
                            days_inventory_left = self.cfg.time_horizon - t - 1

                        schedule.append({
                            'Store': f"Location_{s}",
                            'Product': f"SKU_{p}",
                            'Day': t,
                            'Case_Packs': col['orders'][t],
                            'Units': units_ordered,
                            'Days_Inventory_Left': round(days_inventory_left, 1)
                        })

        
        return pd.DataFrame(schedule)


# Usage
if __name__ == "__main__":
    cfg = ReplenishmentConfig(num_products=1, num_stores=10, time_horizon=7)
    instance = generate_instance(cfg)
    
    solver = ColumnGenerationSolver(instance)
    schedule = solver.solve()
    
    print("\nFinal Replenishment Schedule:")
    print(schedule.to_markdown(index=False))
    # print("RMP status", LpStatus[solver.solve_rmp()[0]])