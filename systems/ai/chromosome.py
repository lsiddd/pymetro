# ./systems/ai/chromosome.py
from typing import List, Any

class Chromosome:
    """
    Representation of the metro network for the Genetic Algorithm.
    """
    def __init__(self, num_lines: int):
        # List of lists. Each inner list represents a line and contains station IDs.
        # IDs are used instead of explicit object references to allow efficient pickling for ProcessPoolExecutor.
        self.lines: List[List[int]] = [[] for _ in range(num_lines)]
        
        # Resources per line
        self.trains_per_line: List[int] = [0 for _ in range(num_lines)]
        self.carriages_per_line: List[int] = [0 for _ in range(num_lines)]
        
        # Track if the line is closed in a loop.
        self.is_loop: List[bool] = [False for _ in range(num_lines)]

    def copy(self) -> 'Chromosome':
        new_c = Chromosome(len(self.lines))
        # Shallow copy of the inner lists is enough since stations are immutable here
        new_c.lines = [line[:] for line in self.lines]
        new_c.trains_per_line = self.trains_per_line[:]
        new_c.carriages_per_line = self.carriages_per_line[:]
        new_c.is_loop = self.is_loop[:]
        return new_c
        
    def is_valid(self) -> bool:
        """
        No duplicate stations within the same line.
        """
        for line in self.lines:
            if len(line) != len(set(line)):
                return False
        return True

    def repair(self):
        """
        Removes consecutive or duplicate stations within the same line.
        """
        for i in range(len(self.lines)):
            seen = set()
            new_line = []
            for station in self.lines[i]:
                if station not in seen:
                    seen.add(station)
                    new_line.append(station)
            self.lines[i] = new_line
