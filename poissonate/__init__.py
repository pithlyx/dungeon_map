import numpy as np
from scipy.spatial import KDTree


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def poisson_disc_samples(width, height, r, seed=None, k=5, distance=euclidean_distance):
    np.random.seed(seed or np.random.randint(0, 1000000))
    tau = 2 * np.pi
    cellsize = r / np.sqrt(2)

    grid = []
    kd_tree = None

    def grid_coords(p):
        return (p / cellsize).astype(int)

    def fits(p):
        if kd_tree is None:
            return True
        else:
            dist, _ = kd_tree.query(p, k=1)
            return dist > r

    p = np.array([width * np.random.rand(), height * np.random.rand()])
    queue = [p]
    grid.append(p)
    kd_tree = KDTree(grid)

    while queue:
        qi = np.random.randint(len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * np.random.rand()
            d = r * np.sqrt(3 * np.random.rand() + 1)
            px, py = qx + d * np.cos(alpha), qy + d * np.sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = np.array([px, py])
            if not fits(p):
                continue
            queue.append(p)
            grid.append(p)
            kd_tree = KDTree(grid)
    return grid


# Test the function
if __name__ == "__main__":
    width, height = 100, 100  # Width and height of the area
    r = 25  # Minimum distance between points
    samples = np.array(poisson_disc_samples(width, height, r)).astype(int)
    print(samples)
