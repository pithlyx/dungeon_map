from proc_map import ProcMap
import hashlib
import numpy as np

class ChunkMap:
    def __init__(self):
        self.seed = np.random.randint(0, 1000000)
        self.chunks = {}
        self.chunks[(0,0)] = Chunk(0, 0, self.seed, 0)
    
    def get_chunk(self, x, y):
        if (x, y) not in self.chunks:
            chunk_id = len(self.chunks)+1
            self.chunks[(x, y)] = Chunk(x, y, self.seed, chunk_id)
        return self.chunks[(x, y)]
    
    def display_chunks(self, chunk, radius):
        cx, cy = chunk.x, chunk.y
        
        for y in range(cy+radius, cy-radius-1, -1):
            for x in range(cx-radius, cx+radius+1):
                if (x, y) in self.chunks:
                    print('[ ]', end=' ')
                else:
                    print('|||', end=' ')
            print()  # Move to the next line after each row

class Chunk:
    def __init__(self, x, y, seed, chunk_id):
        self.x = x
        self.y = y
        self.chunk_id = chunk_id
        self.seed = self.calc_seed(seed)
        self.room_map = ProcMap(seed=self.seed, width=100, height=100, min_distance=20, min_nodes=9, max_nodes=13, k=5, angle_threshold=20, area_threshold=100)
        self.room_map.gen_map()
        
    def calc_seed(self, seed):
        self.x = self.x + 1000 if self.x < -1000 else self.x - 1000 if self.x > 1000 else self.x
        self.y = self.y + 1000 if self.y < -1000 else self.y - 1000 if self.y > 1000 else self.y
        unique_integer = ((self.x + 1000) * 2001) + (self.y + 1000)
        print(f'x: {self.x}, y: {self.y}, unique_integer: {unique_integer}')
        seed = seed+unique_integer
        print(f'seed: {seed}')
        return seed
            
class Player:
    directions = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
    def __init__(self, username):
        self.username = username
        self.map = ChunkMap()
        self.chunk = self.map.get_chunk(0, 0)
        self.x, self.y = None,None
    
    def move_chunk(self, direction):
        if direction not in self.directions:
            raise ValueError(f'Invalid direction: {direction}')
        d = self.directions[direction]
        # Update the chunk coordinates based on the direction
        self.chunk = self.map.get_chunk(self.chunk.x + d[0], self.chunk.y + d[1])
        self.chunk.room_map.plot(mst=True,extra=True, rooms=True, save=True, filename="current_chunk.png")
        self.map.display_chunks(player.chunk, 2)
        print('-'*80)

def menu():
    player = Player('test')
    while True:
        direction = input('Enter a direction: ')
        try:
            player.move_chunk(direction)
        except ValueError as e:
            print(e)

menu()