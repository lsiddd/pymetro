import pygame

class River:
    def __init__(self, points):
        """
        Initialize river with list of points
        Points should be in format [{'x': x1, 'y': y1}, {'x': x2, 'y': y2}, ...]
        """
        self.points = points
    
    def contains(self, x, y):
        """Check if point (x, y) is inside the river polygon"""
        # Ray casting algorithm for point-in-polygon test
        n = len(self.points)
        inside = False
        
        p1x, p1y = self.points[0]['x'], self.points[0]['y']
        for i in range(1, n + 1):
            p2x, p2y = self.points[i % n]['x'], self.points[i % n]['y']
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def draw(self, screen):
        """Draw the river on screen"""
        if len(self.points) < 3:
            return
        
        # Convert points to pygame format
        pygame_points = [(point['x'], point['y']) for point in self.points]
        
        # Draw river as filled polygon
        pygame.draw.polygon(screen, (173, 216, 230), pygame_points)  # Light blue
        pygame.draw.polygon(screen, (135, 206, 250), pygame_points, 2)  # Darker blue border