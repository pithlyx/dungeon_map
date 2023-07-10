# MSTMap

## Description:

This is a Python class that creates a graph representation of a map, using a minimum spanning tree (MST) algorithm for the generation of its base structure and an additional random edges generation for further detail.

This class relies on the `numpy`, `scipy`, `poissonate`, `networkx`, `collections`, and `matplotlib` packages.

## Class Structure:

The MSTMap class is initialized with parameters: height, width, min_dist, max_edge_factor, and seed (optional). The height and width define the size of the map, min_dist sets the minimum distance between points/nodes on the map, max_edge_factor limits the maximum edge length, and the seed parameter is used for random number generation, enabling reproducibility.

Internally, the class uses several data structures including lists and dictionaries to store the nodes, edges, and other attributes of the map graph.

## Methods:

1. **generate_random_edges(self, average_percentage, starting_weight):** This method is used to generate additional edges on top of the MST. The number of edges added is determined by average_percentage, which should be between 0 and 1. The method ensures the percentage of additional edges is within 10% of the given average_percentage. The starting_weight parameter, which should also be between 0 and 1, determines the initial probability of an edge being added.

2. **generate_samples(self):** This method uses Poisson Disc Sampling to generate a set of points that are used as nodes in the graph. The points are generated in such a way that no two points are closer together than a specified minimum distance.

3. **create_graph(self):** This method creates a complete graph from the generated samples using Delaunay triangulation.

4. **generate_mst(self):** This method generates the Minimum Spanning Tree (MST) from the complete graph. It uses the Kruskal's algorithm to find the MST. If less than 4 points are found initially, new seeds are generated until enough points are found.

5. **\_update_full_map(self):** This is a helper function that updates the 'full_map' attribute after the MST and the additional random edges have been created. The 'full_map' attribute is a nested dictionary where each key is a node and the value is another dictionary whose keys are adjacent nodes and values are the distances to these nodes.

6. **save_final_map(self, file_path='final_map.png'):** This method saves the final map as a PNG image. Nodes are displayed as red points and edges are displayed as blue lines.

## Example Usage:

To create a new map with a specified height, width, and minimum distance between nodes, you would do the following:

```python
my_map = MSTMap(100, 100, 20, seed=42)
my_map.generate_mst()
my_map.generate_random_edges(0.2, 0.1)
my_map.save_final_map()
```

This would create a new map of size 100x100, with a minimum distance of 20 between nodes. A seed of 42 is used for reproducibility. The map would first generate an MST, then add additional random edges (approximately 20% of the total possible edges), and finally save the image of the map to a file 'final_map.png'.
