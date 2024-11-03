import heapq
import math

def read_input(file_path):
    with open(file_path, 'r') as f:
        start_x, start_y, goal_x, goal_y = map(int, f.readline().split())
        grid = [list(map(int, line.split())) for line in f]
    return (start_y, start_x), (goal_y, goal_x), grid

def heuristic(a, b):
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def get_neighbors(node, grid):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    neighbors = []
    for dy, dx in directions:
        ny, nx = node[0] + dy, node[1] + dx
        if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and grid[ny][nx] != 1:
            neighbors.append((ny, nx))
    return neighbors

def get_direction(a, b):
    dy, dx = b[0] - a[0], b[1] - a[1]
    if dy == -1 and dx == 0: return 'N'
    if dy == 1 and dx == 0: return 'S'
    if dy == 0 and dx == 1: return 'E'
    if dy == 0 and dx == -1: return 'W'
    if dy == -1 and dx == 1: return 'NE'
    if dy == -1 and dx == -1: return 'NW'
    if dy == 1 and dx == 1: return 'SE'
    if dy == 1 and dx == -1: return 'SW'

def astar(start, goal, grid):
    open_list = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()
    
    while open_list:
        current = heapq.heappop(open_list)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], len(closed_set), [f_score[node] for node in path[::-1]]
        
        closed_set.add(current)
        
        for neighbor in get_neighbors(current, grid):
            if neighbor in closed_set:
                continue
            
            tentative_g_score = g_score[current] + (1 if neighbor[0] == current[0] or neighbor[1] == current[1] else math.sqrt(2))
            
            if neighbor not in [i[1] for i in open_list]:
                heapq.heappush(open_list, (f_score.get(neighbor, float('inf')), neighbor))
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
    
    return None, len(closed_set), []

def format_output(path, nodes_expanded, f_values, grid):
    output = f"{len(path) - 1}\n"
    directions = [get_direction(path[i], path[i+1]) for i in range(len(path)-1)]
    output += ' '.join(directions) + "\n"
    output += f"{nodes_expanded}\n"
    output += ' '.join(f"{f:.1f}" for f in f_values) + "\n"
    
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if (y, x) in path:
                if (y, x) == path[0]:
                    output += "2 "
                elif (y, x) == path[-1]:
                    output += "5 "
                else:
                    output += "4 "
            else:
                output += f"{cell} "
        output += "\n"
    
    return output.strip()

def main():
    input_file = "/Users/tewoflosgirmay/Desktop/AI Project/CS 4613 AI F24 Input1.txt"
    start, goal, grid = read_input(input_file)
    path, nodes_expanded, f_values = astar(start, goal, grid)
    output = format_output(path, nodes_expanded, f_values, grid)
    print(output)

if __name__ == "__main__":
    main()