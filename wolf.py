
import random
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. INITIALIZATION
# ==========================================

random.seed(42)

N = 30
MAX_ITER = 300

P = 6000
L = 14
E = 30 * (10 ** 6)
G = 12 * (10 ** 6)

# Design variables: h, l, t, b
# h = Weld thickness
# l = Weld length
# t = Beam height
# b = Beam thickness

LOWER = [0.1, 0.1, 0.1, 0.1]
UPPER = [2.0, 10.0, 10.0, 2.0]


# ==========================================
# 2. OBJECTIVE FUNCTION
# ==========================================

def objective(x):

    h, l, t, b = x

    cost = (
        1.10471 * h**2 * l
        + 0.04811 * t * b * (14 + l)
    )

    return cost


# ==========================================
# 3. ENGINEERING CONSTRAINTS
# ==========================================

def constraints(x):

    h, l, t, b = x

    # Primary shear stress
    tau_prime = P / (math.sqrt(2) * h * l)

    # Bending moment
    M = P * (L + l / 2)

    # Radius
    R = math.sqrt(
        (l**2 / 4) + ((h + t) / 2)**2
    )

    # Polar moment of inertia
    J = 2 * math.sqrt(2) * h * l * (
        (l**2 / 12) + ((h + t) / 2)**2
    )

    # Secondary shear stress
    tau_double = M * R / J

    # Resultant shear stress
    tau = math.sqrt(
        tau_prime**2
        + 2 * tau_prime * tau_double * l / (2 * R)
        + tau_double**2
    )

    # Bending stress
    sigma = (6 * P * L) / (b * t**2)

    # Deflection
    delta = (4 * P * L**3) / (E * b * t**3)

    # Critical buckling load
    Pc = (
        4.013 * E *
        math.sqrt((t**2 * b**6) / 36) / L**2
    ) * (
        1 - (t / (2 * L)) *
        math.sqrt(E / (4 * G))
    )

    # Constraint equations: all must be <= 0

    g = [
        tau - 13600,
        sigma - 30000,
        h - b,
        0.125 - h,
        delta - 0.25,
        P - Pc
    ]

    return g


# ==========================================
# 4. FITNESS EVALUATION
# ==========================================

def evaluate(x):

    cost = objective(x)
    g = constraints(x)

    feasible = all(value <= 1e-8 for value in g)

    violation = sum(
        max(0, value)**2 for value in g
    )

    return feasible, violation, cost


# ==========================================
# 5. SELECT ALPHA, BETA AND DELTA
# ==========================================

def select_leaders(wolves):

    wolves.sort(
        key=lambda x: (
            0 if evaluate(x)[0] else 1,
            evaluate(x)[1] if not evaluate(x)[0]
            else evaluate(x)[2]
        )
    )

    alpha = wolves[0][:]
    beta = wolves[1][:]
    delta = wolves[2][:]

    return alpha, beta, delta


# ==========================================
# 6. INITIALIZE WOLF POPULATION
# ==========================================

wolves = []

for i in range(N):

    wolf = []

    for j in range(4):

        position = random.uniform(
            LOWER[j], UPPER[j]
        )

        wolf.append(position)

    wolves.append(wolf)


# Select initial leaders
alpha, beta, delta = select_leaders(wolves)


# ==========================================
# 7. TRACK BEST SOLUTION
# ==========================================

best_cost = float('inf')
best_solution = None

convergence_curve = []


# ==========================================
# 8. MAIN GWO ALGORITHM
# ==========================================

for iteration in range(MAX_ITER):

    # Decrease a from 2 to 0
    a = 2 - (2 * iteration / MAX_ITER)

    # Update positions of all wolves

    for i in range(N):

        X = wolves[i]

        X1 = []
        X2 = []
        X3 = []

        for j in range(4):

            # Alpha coefficients
            r1 = random.random()
            r2 = random.random()

            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            # Beta coefficients
            r1 = random.random()
            r2 = random.random()

            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            # Delta coefficients
            r1 = random.random()
            r2 = random.random()

            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            # Calculate distances
            D1 = abs(C1 * alpha[j] - X[j])
            D2 = abs(C2 * beta[j] - X[j])
            D3 = abs(C3 * delta[j] - X[j])

            # Calculate new positions
            X1.append(alpha[j] - A1 * D1)
            X2.append(beta[j] - A2 * D2)
            X3.append(delta[j] - A3 * D3)

        # Update wolf position
        for j in range(4):

            X[j] = (X1[j] + X2[j] + X3[j]) / 3

            # Boundary correction
            X[j] = max(
                LOWER[j],
                min(X[j], UPPER[j])
            )

    # Recalculate leaders
    alpha, beta, delta = select_leaders(wolves)

    # Track best feasible solution found so far
    for wolf in wolves:

        feasible, violation, cost = evaluate(wolf)

        if feasible and cost < best_cost:

            best_cost = cost
            best_solution = wolf[:]

    # Store convergence value
    if best_solution is not None:
        convergence_curve.append(best_cost)
    else:
        convergence_curve.append(float('nan'))

    # Display progress
    if (iteration + 1) % 50 == 0:

        print(
            "Iteration:", iteration + 1,
            "| Best Cost:", best_cost
        )


# ==========================================
# 9. DISPLAY OPTIMIZED RESULTS
# ==========================================

print("\n====================================")
print(" GWO WELDED BEAM DESIGN OPTIMIZATION")
print("====================================")

if best_solution is not None:

    h, l, t, b = best_solution

    print("\nOPTIMIZED DESIGN VARIABLES")

    print("Weld Thickness (h):", round(h, 6))
    print("Weld Length (l):", round(l, 6))
    print("Beam Height (t):", round(t, 6))
    print("Beam Thickness (b):", round(b, 6))

    print("\nMinimum Manufacturing Cost:",
          round(best_cost, 6))

    print("\nFEASIBILITY CHECK")

    print("Feasible Design: True")

    names = [
        "Shear Stress",
        "Bending Stress",
        "Geometric Constraint (h <= b)",
        "Minimum Weld Thickness",
        "Deflection",
        "Buckling Load"
    ]

    g = constraints(best_solution)

    for name, value in zip(names, g):

        print(
            name, ":",
            "Satisfied" if value <= 1e-8
            else "Violated"
        )

else:

    print("No feasible design found.")
    print("Try increasing population or iterations.")


# ==========================================
# 10. PLOT CONVERGENCE GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, MAX_ITER + 1),
    convergence_curve,
    linewidth=2,
    label="GWO Best Cost"
)

plt.xlabel("Number of Iterations")
plt.ylabel("Best Manufacturing Cost")

plt.title(
    "GWO Convergence Curve - Welded Beam Design"
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# ==========================================
# 11. DISPLAY FINAL DESIGN SUMMARY
# ==========================================

if best_solution is not None:

    print("\nFinal Result:")
    print("Best Solution =", best_solution)
    print("Best Cost =", best_cost)
