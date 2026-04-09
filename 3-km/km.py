import numpy as np

class KM:
    def __init__(self, config: dict, data: np.ndarray):
        self._seed = config['seed']
        self._clusters = config['clusters']
        self._data = data
        self._centroid_shift = config['centroid_shift']
        self._cycles = config['cycles']
        self._generate_seed_points()
        self._loop()

    def _generate_seed_points(self):
        """Initialize centroids by randomly selecting points from the dataset."""
        print("Generating initial centroids...")

        # Calculate bounds of the dataset
        max_x = np.max(self._data[:, 0])
        min_x = np.min(self._data[:, 0])
        max_y = np.max(self._data[:, 1])
        min_y = np.min(self._data[:, 1])
        self._bounds = (min_x, max_x, min_y, max_y)

        # Calculate the size and density of each macroblock
        size_x = (max_x - min_x) / self._clusters
        size_y = (max_y - min_y) / self._clusters
        macroblocks = self._clusters * self._clusters
        density = len(self._data) / macroblocks

        # Build list of potential centroids with a minimum density
        potential_centroids = []
        for i in range(self._clusters):
            low_x = min_x + (i * size_x)
            high_x = low_x + size_x
            mid_x = (low_x + high_x) / 2
            for j in range(self._clusters):
                low_y = min_y + (j * size_y)
                high_y = low_y + size_y
                mid_y = (low_y + high_y) / 2

                # Append centroid if it has a high enough density
                mask = (
                    (self._data[:, 0] >= low_x) &
                    (self._data[:, 0] <= high_x) &
                    (self._data[:, 1] >= low_y) &
                    (self._data[:, 1] <= high_y)
                )
                points = self._data[mask]

                if len(points) > density:
                    potential_centroids.append((mid_x, mid_y))

        # Randomly select centroids from the potential list
        rng = np.random.default_rng(self._seed)
        centroids = rng.choice(potential_centroids, self._clusters, replace=False)

        # Calculate the initial radius of each cluster
        radius = None
        for i in range(self._clusters):
            for j in range(i + 1, self._clusters):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if radius is None or dist < 2 * radius:
                    radius = dist / 2

        self._centroids = [(centroid, radius) for centroid in centroids]
        print(f"Generated {self._clusters} initial centroids...")
        for centroid, radius in self._centroids:
            print(f"({centroid[0]:.2f}, {centroid[1]:.2f}) - {radius:.2f}")
        print("-"*30)

    def _loop(self):
        """Main K-Means loop."""
        clustered_data = []
        for i in range(self._cycles):
            # Start off with all data points marked as outliers (-1)
            clustered_data = [(datapoint, -1) for datapoint in self._data]
            unstable = False

            # Assign each data point to the cluster they belong to
            for centroid_idx in range(len(self._centroids)):
                centroid = self._centroids[centroid_idx]
                for clustered_datapoint_idx in range(len(clustered_data)):
                    clustered_datapoint = clustered_data[clustered_datapoint_idx]

                    # Skip if the datapoint is already assigned to a cluster
                    if clustered_datapoint[1] != -1:
                        continue

                    # Add datapoint to cluster if it is in the radius of the current centroid
                    dist = np.linalg.norm(centroid[0] - clustered_datapoint[0])
                    if dist < centroid[1]:
                        clustered_data[clustered_datapoint_idx] = (clustered_datapoint[0], centroid_idx)

                # Move centroid to the mean of the points in the cluster
                cluster_points = [clustered_datapoint[0] for clustered_datapoint in clustered_data if clustered_datapoint[1] == centroid_idx]
                x_new = np.mean([cluster_point[0] for cluster_point in cluster_points])
                y_new = np.mean([cluster_point[1] for cluster_point in cluster_points])

                # If the centroid moved too much, we need to continue the loop
                centroid_shift = np.linalg.norm(centroid[0] - np.array([x_new, y_new]))
                if centroid_shift > self._centroid_shift:
                    unstable = True

                self._centroids[centroid_idx] = ((x_new, y_new), centroid[1])
                print(f"Centroid {centroid_idx} moved to ({x_new:.2f}, {y_new:.2f})")

            outliers = [clustered_datapoint for clustered_datapoint in clustered_data if clustered_datapoint[1] == -1]
            self._print_loop(i+1, len(outliers))
            if not unstable:
                break

        self._results = clustered_data

    def _print_loop(self, loop: int, outliers: int):
        """Prints the results of a single loop."""
        print(f"Loop {loop} completed: {outliers} outliers found:")
        for centroid, radius in self._centroids:
            print(f"({centroid[0]:.2f}, {centroid[1]:.2f}) - {radius:.2f}")
        print("-"*30)

    def guess(self, datapoints: np.ndarray):
        """Assigns datapoints to the cluster it belongs to."""
        results = []
        for datapoint in datapoints:
            group = -1
            for i in range(len(self._centroids)):
                centroid, radius = self._centroids[i]
                dist = np.linalg.norm(centroid - datapoint)
                if dist < radius and group == -1:
                    group = i
                    break
            results.append(group)
        return results



    def get_bounds(self):
        return self._bounds

    def get_centroids(self):
        return self._centroids

    def get_results(self):
        return self._results