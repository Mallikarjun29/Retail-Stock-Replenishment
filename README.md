# Retail Stock Replenishment

This project implements a retail stock replenishment system using column generation and linear programming techniques. The goal is to optimize the ordering and inventory management for multiple products across multiple stores over a given time horizon.

## Project Structure

```
Retail-Stock-Replenishment/
│
├── [`Retail-Stock-Replenishment/column_generation.py`](Retail-Stock-Replenishment/column_generation.py )       # Column generation solver implementation
├── [`Retail-Stock-Replenishment/generate_instances.py`](Retail-Stock-Replenishment/generate_instances.py )      # Instance generation for replenishment problem
├── LICENSE                    # License file
├── [`Retail-Stock-Replenishment/master_problem.py`](Retail-Stock-Replenishment/master_problem.py )          # Master problem solver implementation
├── README.md                  # Project documentation
└── __pycache__/               # Compiled Python files
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Retail-Stock-Replenishment.git
    cd Retail-Stock-Replenishment
    ```

2. Install the required dependencies:
    ```sh
    pip install numpy==2.2.4 pandas==2.2.3 scipy==1.15.2 PuLP==3.0.2 networkx==3.4.2
    ```

## Usage

### Generating Instances

The `generate_instances.py` script generates random instances of the replenishment problem based on configurable parameters.

Example usage:
```python
from generate_instances import ReplenishmentConfig, generate_instance

cfg = ReplenishmentConfig(num_products=3, num_stores=2, time_horizon=3)
instance = generate_instance(cfg)
print(instance)
```

### Solving the Master Problem

The `master_problem.py` script solves the master problem using linear programming.

Example usage:
```python
from master_problem import solve_rmp
from generate_instances import ReplenishmentConfig, generate_instance

cfg = ReplenishmentConfig(num_products=3, num_stores=2, time_horizon=3)
instance = generate_instance(cfg)
results_df = solve_rmp(instance)
print(results_df)
```

### Column Generation Solver

The `column_generation.py` script implements the column generation approach to solve the replenishment problem.

Example usage:
```python
from column_generation import ColumnGenerationSolver
from generate_instances import ReplenishmentConfig, generate_instance

cfg = ReplenishmentConfig(num_products=1, num_stores=10, time_horizon=7)
instance = generate_instance(cfg)

solver = ColumnGenerationSolver(instance)
schedule = solver.solve()
print(schedule)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.