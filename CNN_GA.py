import random

# Road network
graph = {
    'A': {'B': 5, 'C': 2},
    'B': {'A': 5, 'C': 1, 'D': 3},
    'C': {'A': 2, 'B': 1, 'D': 4},
    'D': {'B': 3, 'C': 4}
}

# Generate possible routes
routes = [
    ['A', 'B', 'D'],
    ['A', 'C', 'D'],
    ['A', 'B', 'C', 'D']
]

# Calculate route cost
def fitness(route):
    cost = 0

    for i in range(len(route) - 1):
        cost += graph[route[i]][route[i + 1]]

    return cost


# Genetic Algorithm
population = routes.copy()

for generation in range(10):

    # Sort routes by cost (lower is better)
    population.sort(key=fitness)

    # Select best two routes
    parent1 = population[0]
    parent2 = population[1]

    # Crossover
    child = ['A', 'C', 'D']

    # Mutation
    if random.random() < 0.1:
        child = ['A', 'B', 'D']

    # Add child to population
    population.append(child)

    # Keep best 3 routes
    population = sorted(population, key=fitness)[:3]


# Best route
best_route = min(population, key=fitness)

print("Best Route:", " -> ".join(best_route))
print("Route Cost:", fitness(best_route))
