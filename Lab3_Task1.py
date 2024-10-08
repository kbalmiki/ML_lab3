import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# file path
file_path = "inc_vs_rent.csv"

# Load the data
df = pd.read_csv("inc_vs_rent.csv")

# Display the first two columns
print(df.head(10))

# Select the last two columns
x = df.iloc[:, -2]  # Second-to-last column
y = df.iloc[:, -1]  # Last column

# Create scatter plot
plt.scatter(x, y)

# Labeling the axes
plt.xlabel(df.columns[-2])
plt.ylabel(df.columns[-1])

# Show plot
plt.title('Scatter Plot of Income vs Rent')
plt.show()

# Create a new DataFrame from the last two columns of the original df
new_df = df.iloc[:, -2:]  # Selects the second-to-last and last column

# Display the new DataFrame
print(new_df)

# Convert DataFrame to NumPy array
data = new_df.values


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def initialize_centroids(data, k):
    # Randomly select k data points as the initial centroids
    np.random.seed(42)  # Set seed for reproducibility
    return data[np.random.choice(data.shape[0], k, replace=False)]


def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)  # Assign to the nearest centroid
        clusters.append(cluster)
    return np.array(clusters)


def update_centroids(data, clusters, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points_in_cluster = data[clusters == i]
        if len(points_in_cluster) > 0:
            new_centroids[i] = np.mean(points_in_cluster, axis=0)
    return new_centroids


def kmeans(data, k, iterations=10):
    # Step 1: Initialize centroids
    centroids = initialize_centroids(data, k)

    for i in range(iterations):
        # Step 2: Assign points to the closest centroid
        clusters = assign_clusters(data, centroids)

        # Step 3: Calculate new centroids from the mean of points in each cluster
        new_centroids = update_centroids(data, clusters, k)

        # Check for convergence (if centroids do not change, stop)
        if np.all(centroids == new_centroids):
            print(f"Converged after {i + 1} iterations")
            break

        centroids = new_centroids

    return centroids, clusters


# Example usage
k = 2  # Number of clusters
iterations = 10  # Maximum number of iterations

centroids, clusters = kmeans(data, k, iterations)


# Plotting the clusters with the same color and two different markers
plt.figure(figsize=(8, 6))

# Scatter plot with same color and different markers
plt.scatter(data[:, 0], data[:, 1], c='blue', s=100, marker='o', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x', label='Centroids')

# Title and labels
plt.title(f'K-Means Clustering with {k} Clusters with same color')
plt.xlabel(df.columns[-2])
plt.ylabel(df.columns[-1])
plt.legend()
plt.show()

# Plotting the results with different colors for each cluster
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='cool', marker='o', s=100, label='Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title(f'K-Means Clustering (k={k}) clusters with different colors')
plt.xlabel(df.columns[-2])
plt.ylabel(df.columns[-1])
plt.legend()
plt.show()
