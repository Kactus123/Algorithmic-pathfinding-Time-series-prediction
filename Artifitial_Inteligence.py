import pygame
import math
import sys
import random

# Initialize Pygame
pygame.init()
pygame.font.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 800
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Drone Delivery Pathfinding')

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
LIGHT_BLUE = (135, 206, 250)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)  # Color for the hospital
DARK_BLUE = (0, 0, 139)  # Dark blue for the path

# Define font
font = pygame.font.SysFont('Arial', 20)

# Grid dimensions
rows, cols = 30, 30
grid = []

class Node:
    """Represents a node in the grid."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = 'empty'  # Can be 'empty', 'wall', 'start', 'end', 'path', 'open', 'closed', 'hospital'
        self.parent = None  # To trace the path
        self.g = float('inf')  # Cost to move from the start node to this node.
        self.h = float('inf')  # Heuristic - estimated cost from this node to the end node.
        self.f = float('inf')  # Total cost of the node.
        self.traffic = random.randint(1, 5)  # Traffic condition value (1-5)
        self.urgency = random.randint(1, 5)  # Delivery urgency value (1-5)

    def get_neighbors(self, grid):
        """Returns the walkable neighboring nodes."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check all four directions
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                neighbors.append(grid[ny][nx])
        return neighbors

    def __lt__(self, other):
        return self.f < other.f

def draw_grid(path=[]):
    block_width = WIDTH // cols
    block_height = (HEIGHT - 40) // rows

    # Draw all the nodes with their respective colors
    for y in range(rows):
        for x in range(cols):
            rect = pygame.Rect(x * block_width, y * block_height + 40, block_width, block_height)
            node = grid[y][x]
            color = WHITE
            if node.type == 'wall':
                color = BLACK
            elif node.type == 'start':
                color = BLUE
            elif node.type == 'end':
                color = DARK_GREEN
            elif node.type == 'hospital':
                color = YELLOW
            elif node.type == 'path':
                color = LIGHT_BLUE
            elif node.type == 'open':
                color = LIGHT_GREEN
            elif node.type == 'closed':
                color = GRAY
            pygame.draw.rect(display, color, rect)
            pygame.draw.rect(display, GRAY, rect, 1)

    # Draw the path with lines
    if path:
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            x1 = node1.x * block_width + block_width // 2
            y1 = node1.y * block_height + block_height // 2 + 40
            x2 = node2.x * block_width + block_width // 2
            y2 = node2.y * block_height + block_height // 2 + 40
            pygame.draw.line(display, DARK_BLUE, (x1, y1), (x2, y2), 5)

def reset_grid():
    for y in range(rows):
        row = []
        for x in range(cols):
            row.append(Node(x, y))
        grid.append(row)

def heuristic(node, end):
    """Calculate heuristic with adjustments for traffic and urgency."""
    direct_distance = math.sqrt((node.x - end.x) ** 2 + (node.y - end.y) ** 2)
    traffic_factor = 1 + (node.traffic - 1) / 4
    urgency_factor = 1 + (node.urgency - 1) / 4
    return direct_distance * traffic_factor * urgency_factor

def a_star(start, end):
    """Execute A* algorithm to find the shortest path from start to end."""
    open_set = [start]
    start.g = 0
    start.f = heuristic(start, end)

    while open_set:
        current = min(open_set, key=lambda o: o.f)
        if current == end:
            return reconstruct_path(current)

        open_set.remove(current)
        current.type = 'closed'

        for neighbor in current.get_neighbors(grid):
            if neighbor.type == 'wall' or neighbor.type == 'closed':
                continue

            tentative_g_score = current.g + heuristic(current, neighbor)

            if tentative_g_score < neighbor.g:
                neighbor.parent = current
                neighbor.g = tentative_g_score
                neighbor.f = neighbor.g + heuristic(neighbor, end)
                if neighbor.type != 'start' and neighbor.type != 'end' and neighbor.type != 'hospital':
                    neighbor.type = 'path'
                if neighbor not in open_set:
                    open_set.append(neighbor)

        draw_grid()
        pygame.display.update()

    return {}, []

def reconstruct_path(end_node):
    """Reconstruct the path found by A* and calculate path metrics including total cost and distance in km."""
    path = []
    total_cost = 0
    total_distance = 0
    total_traffic = 0
    total_urgency = 0

    current = end_node
    while current.parent is not None:
        path.append(current)
        step_distance = math.sqrt((current.x - current.parent.x) ** 2 + (current.y - current.parent.y) ** 2)
        total_cost += step_distance
        total_distance += step_distance
        total_traffic += current.traffic
        total_urgency += current.urgency
        current = current.parent

    path.append(current)  # Add the start node to the path
    path.reverse()

    distance_in_km = total_distance * 0.1

    results = {
        "Total Cost": total_cost,
        "Total Distance": distance_in_km,
        "Average Traffic": total_traffic / len(path),
        "Average Urgency": total_urgency / len(path)
    }

    return results, path

def main():
    run = True
    clock = pygame.time.Clock()
    start, end, hospital = None, None, None

    reset_grid()

    while run:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # Left click
                x, y = pygame.mouse.get_pos()
                grid_x = x // (WIDTH // cols)
                grid_y = (y - 40) // ((HEIGHT - 40) // rows)
                node = grid[grid_y][grid_x]
                if not start and node not in [end, hospital]:
                    start = node
                    start.type = 'start'
                elif not end and node not in [start, hospital]:
                    end = node
                    end.type = 'end'
                elif not hospital and node not in [start, end]:
                    hospital = node
                    hospital.type = 'hospital'
                elif node not in [start, end, hospital]:
                    node.type = 'wall'

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end and hospital:
                    # Reset the grid except for start, end, walls, and hospital
                    for row in grid:
                        for node in row:
                            if node.type in ['open', 'closed', 'path']:
                                node.type = 'empty'
                            node.parent = None  # Reset the parent
                            node.g = float('inf')
                            node.f = float('inf')

                    # List of potential destinations
                    destinations = [(end, 'Warehouse'), (hospital, 'Hospital')]
                    # Sort destinations based on distance to the start
                    destinations.sort(key=lambda x: heuristic(start, x[0]))

                    # First path from start to the closest destination
                    closest_destination, closest_label = destinations[0]
                    print(f"Path from Central Hub to {closest_label}:")
                    results, path_a_to_b = a_star(start, closest_destination)
                    if results:
                        print(f"Total Cost: {results['Total Cost']:.2f}")
                        print(f"Total Distance: {results['Total Distance']:.2f} km")
                        print(f"Average Traffic: {results['Average Traffic']:.2f}")
                        print(f"Average Urgency: {results['Average Urgency']:.2f}")
                    else:
                        print("No path found.")
                    draw_grid(path_a_to_b)  # Draw the grid with the path from A to B

                    # Second path from the closest destination to the next
                    next_destination, next_label = destinations[1]
                    print(f"Path from Central Hub to {next_label}:")
                    results, path_b_to_c = a_star(closest_destination, next_destination)
                    if results:
                        print(f"Total Cost: {results['Total Cost']:.2f}")
                        print(f"Total Distance: {results['Total Distance']:.2f} km")
                        print(f"Average Traffic: {results['Average Traffic']:.2f}")
                        print(f"Average Urgency: {results['Average Urgency']:.2f}")
                    else:
                        print("No path found.")
                    # Draw the combined path from A to B to C
                    draw_grid(path_a_to_b + path_b_to_c[1:])  # Combine paths for drawing

        draw_grid()  # Ensure the latest grid state is drawn
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
