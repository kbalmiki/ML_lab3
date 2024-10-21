import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)

# file path
file_path = "inc_vs_rent.csv"

# Load the data
df = pd.read_csv(file_path)

# Create a new DataFrame from the last two columns of the original df
new_df = df.iloc[:, -2:]  # Selects the second-to-last and last column

# Display the new DataFrame
print('Data',new_df)

# Convert DataFrame to NumPy array
data = new_df.values

def find_best_k(X, k_min=2, k_max=10):
    silhouette_scores = {}

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        # Silhouette score is only valid when there are more than 1 cluster
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
            silhouette_scores[k] = score
        else:
            silhouette_scores[k] = -1  # Invalid score for a single cluster

    # Find the k with the highest silhouette score
    best_k = max(silhouette_scores, key=silhouette_scores.get)

    return best_k, silhouette_scores

# Example usage (with some dataset X):
best_k, silhouette_scores = find_best_k(data)
print("Best k:", best_k)
print("Silhouette Scores:", silhouette_scores)



# Function to calculate a(i): average intra-cluster distance for point i
def intra_cluster_distance(X, labels, i):
    # Get the cluster label of point i
    cluster_label = labels[i]

    # Get all points in the same cluster as point i
    same_cluster_points = X[labels == cluster_label]

    # Calculate the distance between point i and all points in the same cluster
    if len(same_cluster_points) == 1:
        return 0  # If the point is the only one in the cluster, intra-cluster distance is 0

    distances = pairwise_distances([X[i]], same_cluster_points)[0]

    # Exclude the distance from point i to itself (which is 0)
    return np.mean(distances[distances != 0])


# Function to calculate b(i): average distance to the closest cluster that point i does not belong to
def inter_cluster_distance(X, labels, i):
    # Get the cluster label of point i
    cluster_label = labels[i]

    # Initialize the minimum average distance to the closest other cluster
    min_avg_distance = np.inf

    # Iterate over all unique clusters except the one that point i belongs to
    for other_cluster_label in np.unique(labels):
        if other_cluster_label != cluster_label:
            # Get all points in this other cluster
            other_cluster_points = X[labels == other_cluster_label]

            # Calculate the distance from point i to all points in the other cluster
            distances = pairwise_distances([X[i]], other_cluster_points)[0]

            # Calculate the average distance to the other cluster
            avg_distance = np.mean(distances)

            # Keep track of the minimum average distance to another cluster
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance

    return min_avg_distance


# Function to calculate silhouette coefficient S(i) for point i
def silhouette_coefficient(X, labels, i):
    # Calculate a(i) and b(i)
    a_i = intra_cluster_distance(X, labels, i)
    b_i = inter_cluster_distance(X, labels, i)

    # Calculate silhouette coefficient for point i
    if a_i == 0 and b_i == 0:
        return 0
    return (b_i - a_i) / max(a_i, b_i)


# Function to calculate the average silhouette coefficient for a dataset and a set of cluster labels
def average_silhouette_score(X, labels):
    silhouette_scores = [silhouette_coefficient(X, labels, i) for i in range(len(X))]
    return np.mean(silhouette_scores)


# Function to iterate over a grid of cluster values (2 to 10) and compute the silhouette score
def grid_search_silhouette(X, min_clusters=2, max_clusters=10):
    silhouette_scores = {}
    for num_clusters in range(min_clusters, max_clusters + 1):
        score = average_silhouette_score(X, num_clusters)
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

# Plot silhouette scores to identify the optimal number of clusters
plot_silhouette_scores(silhouette_scores)


# Function to plot a scatter plot with the clustered data
def plot_clusters(X, k):
    # Apply KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_


    # Create the scatter plot with colors corresponding to clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Centers')

    plt.title(f'Scatter Plot of Data with {k} Clusters')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage (assuming best_k is found from the previous steps):
plot_clusters(data, best_k)


# Function to plot clusters and new data points
def plot_clusters_with_new_points(X, k, new_points):
    # Fit the KMeans model to the data
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_

    # Predict the clusters for the new data points
    new_labels = kmeans.predict(new_points)

    # Plot the original data points with cluster labels
    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)

    # Plot the cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Centers')

    # Plot the new data points and their predicted clusters
    plt.scatter(new_points[:, 0], new_points[:, 1], c=new_labels, cmap='cool', marker='D', s=100,
                label='New Points')

    # Add legend
    plt.legend(loc='best')

    # Title and labels
    plt.title(f'Cluster Prediction for New Data Points with {k} Clusters')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Show the plot
    plt.grid(True)
    plt.show()

    # Print which clusters the new points belong to
    for i, point in enumerate(new_points):
        print(f"New point {point} belongs to cluster {new_labels[i]}.")
# Define the new points
new_points = np.array([[1010, 320.12], [1258, 320], [980, 292.4]])

# Plot the clusters with the new points
plot_clusters_with_new_points(data, best_k, new_points)