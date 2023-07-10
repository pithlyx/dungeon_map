from poissonate import poisson_disc_samples
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from matplotlib.patches import Rectangle

class ProcMap:
    def __init__(self, seed=None, width=100, height=100, min_distance=20, min_nodes=9, max_nodes=13, k=5, angle_threshold=20, area_threshold=100):
        self.seed = seed or np.random.randint(0, 1000000)
        self.width = width
        self.height = height
        self.min_distance = min_distance
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.k = k
        self.points = None
        self.tri = None
        self.mst = None
        self.angle_threshold = math.radians(
            angle_threshold)  # convert to radians
        self.area_threshold = area_threshold
        self.rooms = []

    def gen_map(self):
        while self.points is None or len(self.points) < self.min_nodes or len(self.points) > self.max_nodes:
            np.random.seed(self.seed)
            self.points = np.array(self.generate_points()).astype(int)
            if len(self.points) < self.min_nodes or len(self.points) > self.max_nodes:
                self.seed += np.random.randint(0, 1000000)
        self.tri = self.triangulate()
        self.clean_tris()
        self.gen_mst()
        self.gen_extra_connections(0.2)
        self.gen_rooms(.65)
        self.connect_rooms()
        
    def generate_points(self):
        return poisson_disc_samples(self.width, self.height, self.min_distance, self.seed, self.k)

    def triangulate(self):
        if self.points is not None:
            return Delaunay(self.points)

    def clean_tris(self):
        clean_tris = []
        for tri in self.tri.simplices:
            a, b, c = self.points[tri]
            angles = self.compute_angles(a, b, c)
            area = self.compute_area(a, b, c)
            if min(angles) >= self.angle_threshold and area >= self.area_threshold:
                clean_tris.append(tri)
        self.tri.simplices = np.array(clean_tris)

    @staticmethod
    def compute_angles(a, b, c):
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ac = np.linalg.norm(a - c)
        angles = [math.acos((ab**2 + bc**2 - ac**2) / (2 * ab * bc)),
                  math.acos((bc**2 + ac**2 - ab**2) / (2 * bc * ac)),
                  math.acos((ac**2 + ab**2 - bc**2) / (2 * ac * ab))]
        return angles

    @staticmethod
    def compute_area(a, b, c):
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ac = np.linalg.norm(a - c)
        s = (ab + bc + ac) / 2
        return math.sqrt(s * (s - ab) * (s - bc) * (s - ac))

    def gen_mst(self):
        # Get pairwise distances between points
        dist_matrix = squareform(pdist(self.points))
        # Compute the minimum spanning tree
        self.mst = minimum_spanning_tree(dist_matrix).toarray()

    def gen_extra_connections(self, prob=0.5):
        mst_edges = set()
        for i in range(len(self.mst)):
            for j in range(len(self.mst)):
                if self.mst[i, j] != 0:
                    # Ensure the edge is always (smaller, larger)
                    edge = (i, j) if i < j else (j, i)
                    mst_edges.add(edge)

        self.extra_connections = []
        for tri in self.tri.simplices:
            for i in range(3):
                j = (i + 1) % 3
                edge = (tri[i], tri[j]) if tri[i] < tri[j] else (
                    tri[j], tri[i])
                if edge not in mst_edges and np.random.rand() < prob:
                    dist = np.linalg.norm(
                        self.points[edge[0]] - self.points[edge[1]])
                    self.extra_connections.append((edge, dist))

    def gen_rooms(self, dist_mod=0.5):
        self.rooms = []
        dist_matrix = cdist(self.points, self.points)
        for i, point in enumerate(self.points):
            dists = np.partition(dist_matrix[i], 2)  # Partial sort the distances
            calc_dist = int(dists[1] * dist_mod)  # Second smallest distance is the distance to the closest point
            
            # Make sure the calculated distance is an odd number
            if calc_dist % 2 == 0:
                calc_dist -= 1
            
            # Generate a random odd width and height
            start = int(calc_dist * 0.8)
            if start%2 == 0:
                start+=1
            end = int(calc_dist * 1.2)
            width = np.random.choice(range(start, end, 2))
            height = np.random.choice(range(start, end, 2))


            # Calculate the x and y coordinates of the center of each room
            x = point[0]
            y = point[1]

            new_room = Room(i, x, y, width, height)
            self.rooms.append(new_room)


    
    def connect_rooms(self):
        # Connect rooms according to the MST
        for i in range(len(self.mst)):
            for j in range(len(self.mst)):
                if self.mst[i, j] != 0:
                    self.rooms[i].connect(self.rooms[j])

        # Connect rooms according to the extra connections
        for edge, _ in self.extra_connections:
            self.rooms[edge[0]].connect(self.rooms[edge[1]])

    def plot(self, points=False, tri=False, mst=False, extra=False, rooms=False, save=False, filename="map.png"):
        # sourcery skip: move-assign
        point_color = 'blue'  # Change color here
        tri_color = 'black'  # Change color here
        mst_color = 'coral'  # Change color here
        extra_color = 'purple'  # Change color here
        room_color = 'lightblue'  # Change color here

        if points:
            plt.scatter(*zip(*self.points), color=point_color)

        if tri and self.tri is not None:
            plt.triplot(self.points[:, 0], self.points[:, 1], self.tri.simplices, color=tri_color)

        if mst and self.mst is not None:
            for i in range(len(self.mst)):
                for j in range(len(self.mst)):
                    if self.mst[i, j] != 0:
                        plt.plot([self.points[i, 0], self.points[j, 0]], [self.points[i, 1], self.points[j, 1]], color=mst_color)

        if extra and self.extra_connections is not None:
            for edge, _ in self.extra_connections:
                plt.plot([self.points[edge[0], 0], self.points[edge[1], 0]], [self.points[edge[0], 1], self.points[edge[1], 1]], color=extra_color)

        if rooms and self.rooms is not None:
            for room in self.rooms:
                # Subtract half of the width and height from the x and y coordinates to draw the rectangle centered at (x, y)
                rect_x = room.x - room.width // 2
                rect_y = room.y - room.height // 2

                plt.gca().add_patch(Rectangle((rect_x, rect_y), room.width, room.height, fill=True, color=room_color, linewidth=2))
                # Plotting room information
                info = f"{room.room_id}\n({room.x},{room.y})\n{room.width}x{room.height}"
                plt.text(room.x, room.y, info, color='black', ha='center', va='center')

        if not save:
            plt.show()
        else:
            plt.savefig(filename)
            plt.clf()
            

class Room:
    def __init__(self, room_id, x, y, width, height):
        self.room_id = room_id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.connections = []

    def connect(self, other):
        if other not in self.connections:  # Check if the room is not already connected
            self.connections.append(other)
            other.connections.append(self)


if __name__ == '__main__':
    new_map = ProcMap(seed=367666, width=100, height=100, min_distance=20, min_nodes=9, max_nodes=13, k=5, angle_threshold=20, area_threshold=100)
    print(f'Start Seed: {new_map.seed}')
    new_map.gen_map()
    print(f'Final Seed: {new_map.seed} | Rooms: {len(new_map.rooms)}')
    for room in new_map.rooms:
        print(f'ID: {room.room_id} | (X: {room.x}, Y: {room.y}) | Width: {room.width}, Height: {room.height} | Connections: {len(room.connections)}, {[room.room_id for room in room.connections]}')

    new_map.plot(points=True, tri=True, save=True, filename="plots/tri.png")
    new_map.plot(points=True, mst=True, save=True, filename="plots/mst.png")
    new_map.plot(points=True, mst=True, extra=True, save=True, filename="plots/mst-extra.png")
    new_map.plot(mst=True,extra=True, rooms=True, save=True, filename="plots/mst-rooms.png")