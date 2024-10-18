import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.set_option('display.max_columns', None)

# file path
file_path = "inc_vs_rent.csv"

# Load the data
df = pd.read_csv(file_path)

# Create a new DataFrame from the last two columns of the original df
new_df = df.iloc[:, -2:]  # Selects the second-to-last and last column

# Display the new DataFrame
print('Data', new_df)

# Convert DataFrame to NumPy array
X = new_df.values


# Function to calculate silhouette score for a given number of clusters
def silhouette_for_clusters(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    return silhouette_avg


# Function to iterate over a grid of cluster values (2 to 10) and compute the silhouette score
def grid_search_silhouette(X, min_clusters=2, max_clusters=10):
    silhouette_scores = {}
    for num_clusters in range(min_clusters, max_clusters + 1):
        score = silhouette_for_clusters(X, num_clusters)
        silhouette_scores[num_clusters] = score
        print(f"Number of clusters: {num_clusters}, Silhouette Score: {score:.4f}")

    return silhouette_scores


# Function to plot silhouette score vs number of clusters
def plot_silhouette_scores(silhouette_scores):
    clusters = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())

    plt.figure(figsize=(8, 5))
    plt.plot(clusters, scores, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores for Different Cluster Numbers")
    plt.grid(True)
    plt.show()


# Function to visualize the clusters with the optimal number of clusters
def plot_clusters(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # Scatter plot of clusters
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(f"K-Means Clustering with {num_clusters} Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


# Function to plot new data points with the clusters
def plot_new_data_with_clusters(X, new_data, predicted_clusters, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)  # This is the fix: Fit the model on X before making predictions
    labels = kmeans.predict(X)

    # Scatter plot of clusters
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(new_data[:, 0], new_data[:, 1], c='red', marker='x', s=100, label='New Data')
    plt.title(f"K-Means Clustering with {num_clusters} Clusters and New Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.legend()
    plt.show()


# Perform grid search for optimal number of clusters
silhouette_scores = grid_search_silhouette(X, min_clusters=2, max_clusters=10)

# Plot silhouette scores to identify the optimal number of clusters
plot_silhouette_scores(silhouette_scores)

# Find the optimal number of clusters (the one with the highest silhouette score)
optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
print(f"Optimal number of clusters: {optimal_clusters}")

# Plot the clusters using the optimal number of clusters
plot_clusters(X, optimal_clusters)

# Predict new data points
new_data = np.array([[1010, 320.12], [1258, 320], [980, 292.4]])
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X)  # Fit the model on your data
predicted_clusters = kmeans.predict(new_data)

print("New data points belong to the following clusters:", predicted_clusters)

# Plot new data points along with clusters
plot_new_data_with_clusters(X, new_data, predicted_clusters, optimal_clusters)

