
import numpy as np
import matplotlib.pyplot as plt
import random

# =========================================================
# IMPROVED ANT COLONY OPTIMIZATION FOR VEHICLE ROUTING (VRP)
# =========================================================

# -------------------- Problem Data -----------------------

# Coordinates: node 0 is the depot; remaining nodes are customers
coordinates = np.array([
    [50, 50],  # Depot
    [20, 60],
    [30, 20],
    [60, 20],
    [70, 60],
    [40, 70],
    [80, 40],
    [10, 40],
    [50, 10],
    [90, 70],
    [25, 85]
], dtype=float)

# Demand of each node (depot demand is zero)
demands = np.array([0, 10, 15, 12, 8, 6, 14, 9, 11, 7, 13])

VEHICLE_CAPACITY = 35
NUM_VEHICLES = 5

# ACO parameters
NUM_ANTS = 30
NUM_ITERATIONS = 150
ALPHA = 1.0          # Pheromone importance
BETA = 3.0           # Distance heuristic importance
INITIAL_RHO = 0.10   # Initial pheromone evaporation rate
Q = 100.0            # Pheromone deposit constant
ELITE_WEIGHT = 2.0   # Extra weight for the global best solution
LOCAL_UPDATE = 0.05  # Local pheromone update strength

NUM_NODES = len(coordinates)
CUSTOMERS = list(range(1, NUM_NODES))

# ---------------- Distance Matrix ------------------------

distance = np.zeros((NUM_NODES, NUM_NODES))

for i in range(NUM_NODES):
    for j in range(NUM_NODES):
        distance[i, j] = np.linalg.norm(
            coordinates[i] - coordinates[j]
        )

# ---------------- Algorithm Helpers ----------------------

def route_distance(route):
    """Calculate distance of a route including depot returns."""
    return sum(
        distance[route[i], route[i + 1]]
        for i in range(len(route) - 1)
    )


def solution_distance(routes):
    """Total distance traveled by all vehicles."""
    return sum(route_distance(route) for route in routes)


def two_opt(route):
    """Improve a route by reversing segments."""
    best = route[:]
    improved = True

    while improved:
        improved = False

        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue

                candidate = (
                    best[:i]
                    + best[i:j + 1][::-1]
                    + best[j + 1:]
                )

                if route_distance(candidate) < route_distance(best):
                    best = candidate
                    improved = True

    return best


def construct_solution(pheromone, rho):
    """
    Construct a feasible VRP solution using probabilistic
    customer selection and vehicle capacity constraints.
    """
    unvisited = set(CUSTOMERS)
    routes = []

    while unvisited:
        if len(routes) >= NUM_VEHICLES:
            return None

        route = [0]
        current = 0
        remaining_capacity = VEHICLE_CAPACITY

        while True:
            feasible = [
                customer for customer in unvisited
                if demands[customer] <= remaining_capacity
            ]

            if not feasible:
                break

            # Improved heuristic: inverse distance
            heuristic = np.array([
                1.0 / (distance[current, customer] + 1e-10)
                for customer in feasible
            ])

            pheromone_values = np.array([
                pheromone[current, customer]
                for customer in feasible
            ])

            weights = (
                pheromone_values ** ALPHA
                * heuristic ** BETA
            )

            # Numerical stability
            weights = np.maximum(weights, 1e-15)
            probabilities = weights / weights.sum()

            next_customer = random.choices(
                feasible, weights=probabilities, k=1
            )[0]

            route.append(next_customer)
            unvisited.remove(next_customer)
            remaining_capacity -= demands[next_customer]

            # Local pheromone update encourages exploration
            pheromone[current, next_customer] = (
                (1 - LOCAL_UPDATE)
                * pheromone[current, next_customer]
                + LOCAL_UPDATE * rho
            )

            current = next_customer

        route.append(0)
        route = two_opt(route)
        routes.append(route)

    return routes


def update_pheromones(pheromone, routes, rho, best_routes):
    """Evaporate pheromone and reinforce high-quality solutions."""
    pheromone *= (1 - rho)

    # Deposit pheromone from the current solution
    total_length = solution_distance(routes)

    for route in routes:
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            pheromone[a, b] += Q / total_length
            pheromone[b, a] += Q / total_length

    # Elitist reinforcement of the global best solution
    best_length = solution_distance(best_routes)

    for route in best_routes:
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            deposit = ELITE_WEIGHT * Q / best_length
            pheromone[a, b] += deposit
            pheromone[b, a] += deposit

    # Prevent pheromone values from becoming too small or large
    np.clip(pheromone, 1e-6, 1e6, out=pheromone)


# ---------------- Main IACO Algorithm ---------------------

def improved_aco():
    random.seed(42)
    np.random.seed(42)

    pheromone = np.ones((NUM_NODES, NUM_NODES))
    np.fill_diagonal(pheromone, 0)

    best_routes = None
    best_length = float("inf")
    history = []
    stagnation = 0

    for iteration in range(NUM_ITERATIONS):
        # Adaptive evaporation: increase exploration pressure
        # when the global best solution stops improving.
        rho = min(0.50, INITIAL_RHO + 0.01 * stagnation)

        iteration_best = None
        iteration_length = float("inf")

        for ant in range(NUM_ANTS):
            routes = construct_solution(pheromone, rho)

            # Skip solutions requiring too many vehicles
            if routes is None:
                continue

            length = solution_distance(routes)

            if length < iteration_length:
                iteration_length = length
                iteration_best = routes

            if length < best_length:
                best_length = length
                best_routes = [r[:] for r in routes]
                stagnation = 0

        if iteration_best is not None:
            update_pheromones(
                pheromone, iteration_best, rho, best_routes
            )

        if iteration_best is None or best_length <= iteration_length:
            stagnation += 1

        history.append(best_length)

        if (iteration + 1) % 10 == 0:
            print(
                f"Iteration {iteration + 1:3d} | "
                f"Best distance: {best_length:.2f} | "
                f"Vehicles used: {len(best_routes)}"
            )

    return best_routes, best_length, history


# ---------------- Run the Algorithm -----------------------

best_routes, best_length, history = improved_aco()

print("\n========== FINAL RESULT ==========")
print(f"Best total distance: {best_length:.2f}")
print(f"Vehicles used: {len(best_routes)}")

for i, route in enumerate(best_routes, start=1):
    load = sum(demands[node] for node in route)
    print(
        f"Vehicle {i}: {' -> '.join(map(str, route))} "
        f"| Load: {load}/{VEHICLE_CAPACITY} "
        f"| Distance: {route_distance(route):.2f}"
    )

# ---------------- Plot Best Routes ------------------------

plt.figure(figsize=(9, 7))

for i, route in enumerate(best_routes):
    x = [coordinates[node, 0] for node in route]
    y = [coordinates[node, 1] for node in route]

    plt.plot(x, y, marker="o", linewidth=2, label=f"Vehicle {i + 1}")

plt.scatter(
    coordinates[0, 0], coordinates[0, 1],
    marker="s", s=180, label="Depot"
)

for node, (x, y) in enumerate(coordinates):
    plt.annotate(
        str(node), (x, y),
        xytext=(5, 5), textcoords="offset points"
    )

plt.title("Improved Ant Colony Optimization for VRP")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- Convergence Plot ------------------------

plt.figure(figsize=(9, 5))
plt.plot(history, linewidth=2)
plt.title("IACO Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Total Distance")
plt.grid(True)
plt.tight_layout()
plt.show()
