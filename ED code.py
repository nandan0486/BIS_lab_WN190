import numpy as np

# -----------------------------
# 1. Generator data
# -----------------------------

# Fuel cost = a + bP + cP^2
a = np.array([100, 120, 150])
b = np.array([5, 4, 3.5])
c = np.array([0.01, 0.015, 0.02])

Pmin = np.array([50, 50, 50])
Pmax = np.array([200, 200, 150])

demand = 300

# -----------------------------
# 2. PSO parameters
# -----------------------------

num_particles = 30
num_iterations = 100

w = 0.7
c1 = 1.5
c2 = 1.5

# Random initialization
np.random.seed(42)

position = np.random.uniform(
    Pmin, Pmax, (num_particles, 3)
)

velocity = np.zeros((num_particles, 3))


# -----------------------------
# 3. Fitness function
# -----------------------------

def fitness(P):
    # Penalty for not meeting demand
    power_error = abs(np.sum(P) - demand)

    fuel_cost = np.sum(
        a + b * P + c * P**2
    )

    penalty = 10000 * power_error

    return fuel_cost + penalty


# -----------------------------
# 4. Initial best values
# -----------------------------

pbest = position.copy()

pbest_fitness = np.array([
    fitness(p) for p in position
])

best_index = np.argmin(pbest_fitness)

gbest = pbest[best_index].copy()
gbest_fitness = pbest_fitness[best_index]


# -----------------------------
# 5. PSO iterations
# -----------------------------

for iteration in range(num_iterations):

    for i in range(num_particles):

        r1 = np.random.random(3)
        r2 = np.random.random(3)

        # Update velocity
        velocity[i] = (
            w * velocity[i]
            + c1 * r1 * (pbest[i] - position[i])
            + c2 * r2 * (gbest - position[i])
        )

        # Update position
        position[i] = position[i] + velocity[i]

        # Keep generator output within limits
        position[i] = np.clip(
            position[i],
            Pmin,
            Pmax
        )

        # Calculate fitness
        current_fitness = fitness(position[i])

        # Update personal best
        if current_fitness < pbest_fitness[i]:

            pbest[i] = position[i].copy()
            pbest_fitness[i] = current_fitness

        # Update global best
        if current_fitness < gbest_fitness:

            gbest = position[i].copy()
            gbest_fitness = current_fitness


# -----------------------------
# 6. Display result
# -----------------------------

# Adjust final solution to exactly meet demand
gbest[2] = demand - gbest[0] - gbest[1]

# Calculate actual fuel cost
final_cost = np.sum(
    a + b * gbest + c * gbest**2
)

print("Optimal Generator Outputs")
print("--------------------------")
print(f"Generator 1: {gbest[0]:.2f} MW")
print(f"Generator 2: {gbest[1]:.2f} MW")
print(f"Generator 3: {gbest[2]:.2f} MW")

print("--------------------------")
print(f"Total Generation: {np.sum(gbest):.2f} MW")
print(f"Load Demand: {demand:.2f} MW")
print(f"Minimum Fuel Cost: {final_cost:.2f} $/hour")