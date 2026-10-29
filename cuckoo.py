
# Cuckoo Search for SVM Hyperparameter Optimization

import numpy as np
from scipy.special import gamma
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load dataset
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Define fitness function
# Each nest represents [log10(C), log10(gamma)]
def fitness(nest):
    C = 10 ** nest[0]
    g = 10 ** nest[1]

    model = make_pipeline(
        StandardScaler(),
        SVC(C=C, gamma=g)
    )

    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    )

    return 1 - scores.mean()  # Minimize error


# 3. Lévy flight using Mantegna's method
def levy_flight(dim):
    beta = 1.5
    sigma = (
        gamma(1 + beta) * np.sin(np.pi * beta / 2)
        / (gamma((1 + beta) / 2) * beta
           * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)

    return u / (np.abs(v) ** (1 / beta) + 1e-12)


# 4. Cuckoo Search parameters
np.random.seed(42)

n_nests = 10
iterations = 20
pa = 0.25

# Search ranges: log10(C), log10(gamma)
lower = np.array([-2.0, -4.0])
upper = np.array([3.0, 1.0])

# 5. Initialize nests
nests = np.random.uniform(
    lower, upper, size=(n_nests, 2)
)
scores = np.array([fitness(n) for n in nests])

best_idx = np.argmin(scores)
best_nest = nests[best_idx].copy()
best_error = scores[best_idx]

# 6. Main Cuckoo Search loop
for iteration in range(iterations):

    # Generate new solutions using Lévy flights
    for i in range(n_nests):
        step = levy_flight(2)
        candidate = nests[i] + 0.1 * step * (upper - lower)
        candidate = np.clip(candidate, lower, upper)

        candidate_error = fitness(candidate)

        # Replace a randomly chosen nest if improved
        j = np.random.randint(n_nests)

        if candidate_error < scores[j]:
            nests[j] = candidate
            scores[j] = candidate_error

    # Abandon the worst nests
    n_abandon = max(1, int(pa * n_nests))
    worst = np.argsort(scores)[-n_abandon:]

    nests[worst] = np.random.uniform(
        lower, upper, size=(n_abandon, 2)
    )
    scores[worst] = np.array(
        [fitness(n) for n in nests[worst]]
    )

    # Update global best
    idx = np.argmin(scores)

    if scores[idx] < best_error:
        best_nest = nests[idx].copy()
        best_error = scores[idx]

    print(
        f"Iteration {iteration + 1}: "
        f"Best CV accuracy = {(1 - best_error) * 100:.2f}%"
    )

# 7. Evaluate optimized model on unseen test data
best_C = 10 ** best_nest[0]
best_gamma = 10 ** best_nest[1]

optimized_model = make_pipeline(
    StandardScaler(),
    SVC(C=best_C, gamma=best_gamma)
)
optimized_model.fit(X_train, y_train)

test_predictions = optimized_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

# Baseline SVM for comparison
baseline = make_pipeline(
    StandardScaler(),
    SVC()
)
baseline.fit(X_train, y_train)

baseline_accuracy = accuracy_score(
    y_test, baseline.predict(X_test)
)

# 8. Display final results
print("\n----- FINAL RESULTS -----")
print(f"Best C: {best_C:.6f}")
print(f"Best gamma: {best_gamma:.6f}")
print(f"Best CV accuracy: {(1 - best_error) * 100:.2f}%")
print(f"Baseline test accuracy: {baseline_accuracy * 100:.2f}%")
print(f"Optimized test accuracy: {test_accuracy * 100:.2f}%")
