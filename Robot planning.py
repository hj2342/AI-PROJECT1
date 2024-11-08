# import heapq
# import math

# def read_input(file_path):
#     with open(file_path, 'r') as f:
#         start_x, start_y, goal_x, goal_y = map(int, f.readline().split())
#         grid = [list(map(int, line.split())) for line in f]
#     return (start_y, start_x), (goal_y, goal_x), grid

# def heuristic(a, b):
#     return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

# def get_neighbors(node, grid):
#     directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
#     neighbors = []
#     for dy, dx in directions:
#         ny, nx = node[0] + dy, node[1] + dx
#         if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and grid[ny][nx] != 1:
#             neighbors.append((ny, nx))
#     return neighbors

# def get_direction(a, b):
#     dy, dx = b[0] - a[0], b[1] - a[1]
#     if dy == -1 and dx == 0: return 'N'
#     if dy == 1 and dx == 0: return 'S'
#     if dy == 0 and dx == 1: return 'E'
#     if dy == 0 and dx == -1: return 'W'
#     if dy == -1 and dx == 1: return 'NE'
#     if dy == -1 and dx == -1: return 'NW'
#     if dy == 1 and dx == 1: return 'SE'
#     if dy == 1 and dx == -1: return 'SW'

# def astar(start, goal, grid):
#     open_list = [(0, start)]
#     came_from = {}
#     g_score = {start: 0}
#     f_score = {start: heuristic(start, goal)}
#     closed_set = set()
    
#     while open_list:
#         current = heapq.heappop(open_list)[1]
        
#         if current == goal:
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start)
#             return path[::-1], len(closed_set), [f_score[node] for node in path[::-1]]
        
#         closed_set.add(current)
        
#         for neighbor in get_neighbors(current, grid):
#             if neighbor in closed_set:
#                 continue
            
#             tentative_g_score = g_score[current] + (1 if neighbor[0] == current[0] or neighbor[1] == current[1] else math.sqrt(2))
            
#             if neighbor not in [i[1] for i in open_list]:
#                 heapq.heappush(open_list, (f_score.get(neighbor, float('inf')), neighbor))
#             elif tentative_g_score >= g_score.get(neighbor, float('inf')):
#                 continue
            
#             came_from[neighbor] = current
#             g_score[neighbor] = tentative_g_score
#             f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
    
#     return None, len(closed_set), []

# def format_output(path, nodes_expanded, f_values, grid):
#     output = f"{len(path) - 1}\n"
#     directions = [get_direction(path[i], path[i+1]) for i in range(len(path)-1)]
#     output += ' '.join(directions) + "\n"
#     output += f"{nodes_expanded}\n"
#     output += ' '.join(f"{f:.1f}" for f in f_values) + "\n"
    
#     for y, row in enumerate(grid):
#         for x, cell in enumerate(row):
#             if (y, x) in path:
#                 if (y, x) == path[0]:
#                     output += "2 "
#                 elif (y, x) == path[-1]:
#                     output += "5 "
#                 else:
#                     output += "4 "
#             else:
#                 output += f"{cell} "
#         output += "\n"
    
#     return output.strip()

# def main():
#     input_file = "/Users/hariharanjanardhanan/Desktop/__/AI/2__.Proj1/AI-PROJECT1/Sample input.txt"
#     start, goal, grid = read_input(input_file)
#     path, nodes_expanded, f_values = astar(start, goal, grid)
#     output = format_output(path, nodes_expanded, f_values, grid)
#     print(output)

# if __name__ == "__main__":
#     main()


# Trail 1

# import heapq
# import math

# def read_input(file_path):
#     with open(file_path, 'r') as f:
#         start_y, start_x, goal_y, goal_x = map(int, f.readline().split())
#         grid = [list(map(int, line.split())) for line in f]
#     return (start_y, start_x), (goal_y, goal_x), grid

# def heuristic(a, b):
#     dx = abs(a[1] - b[1])
#     dy = abs(a[0] - b[0])
#     return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
# # def heuristic(a, b):
# #     return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
# def get_neighbors(node, grid):
#     directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
#     neighbors = []
#     for i, (dy, dx) in enumerate(directions):
#         ny, nx = node[0] + dy, node[1] + dx
#         if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and grid[ny][nx] != 1:
#             neighbors.append((i, (ny, nx)))
#     print(f"Neighbors of {node}: {neighbors}")

#     return neighbors

# def angle_cost(prev_move, next_move, k):
#     if prev_move is None:
#         return 0
#     angle_diff = abs(next_move - prev_move)
#     if angle_diff > 4:
#         angle_diff = 8 - angle_diff
#     return k * angle_diff / 180

# def distance_cost(move):
#     return math.sqrt(2) if move % 2 == 1 else 1

# # def astar(start, goal, grid, k):
# #     open_list = [(0, start, None)]
# #     came_from = {}
# #     g_score = {start: 0}
# #     f_score = {start: heuristic(start, goal)}
# #     closed_set = set()
# #     nodes_generated = 1
# #     max_iterations = 10000  

# #     while open_list and nodes_generated < max_iterations:
# #         _, current, prev_move = heapq.heappop(open_list)
# #         print(f"Exploring node: {current}, f_score: {f_score[current]}")


# #         if current == goal:
# #             print("Goal reached!")

# #             path = []
# #             actions = []
# #             f_values = [f_score[current]]
# #             while current in came_from:
# #                 path.append(current)
# #                 actions.append(came_from[current][1])
# #                 current = came_from[current][0]
# #                 f_values.append(f_score[current])
# #             path.append(start)
# #             return path[::-1], actions[::-1], f_values[::-1], nodes_generated

# #         closed_set.add(current)

# #         for move, neighbor in get_neighbors(current, grid):
# #             if neighbor in closed_set:
# #                 continue

# #             tentative_g_score = g_score[current] + distance_cost(move) + angle_cost(prev_move, move, k)

# #             if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
# #                 came_from[neighbor] = (current, move)
# #                 g_score[neighbor] = tentative_g_score
# #                 f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
# #                 heapq.heappush(open_list, (f_score[neighbor], neighbor, move))
# #                 nodes_generated += 1
# #                 print(f"  Added neighbor: {neighbor}, f_score: {f_score[neighbor]}")
# #     print(f"No path found after exploring {nodes_generated} nodes.")

# #     return None, None, None, nodes_generated
# def astar(start, goal, grid, k, max_iterations=1000):
#     open_list = [(0, start, None)]
#     came_from = {}
#     g_score = {start: 0}
#     f_score = {start: heuristic(start, goal)}
#     closed_set = set()
#     nodes_generated = 1
#     iterations = 0

#     while open_list and iterations < max_iterations:
#         iterations += 1
#         current_f, current, prev_move = heapq.heappop(open_list)
        
#         if current == goal:
#             print("Goal reached!")
#             path, actions, f_values = reconstruct_path(came_from, current, g_score, f_score)
#             return path, actions, f_values, nodes_generated

#         closed_set.add(current)

#         for move, neighbor in get_neighbors(current, grid):
#             if neighbor in closed_set:
#                 continue

#             tentative_g_score = g_score[current] + distance_cost(move) + angle_cost(prev_move, move, k)

#             if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                 came_from[neighbor] = (current, move)
#                 g_score[neighbor] = tentative_g_score
#                 f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
#                 heapq.heappush(open_list, (f_score[neighbor], neighbor, move))
#                 nodes_generated += 1

#     print(f"No path found after {iterations} iterations and exploring {nodes_generated} nodes.")
#     return None, None, None, nodes_generated

# # def reconstruct_path(came_from, current):
# #     path = []
# #     actions = []
# #     while current in came_from:
# #         path.append(current)
# #         current, action = came_from[current]
# #         actions.append(action)
# #     path.append(current)  # Add the start node
# #     return path[::-1], actions[::-1]
# def reconstruct_path(came_from, current, g_score, f_score):
#     path = []
#     actions = []
#     f_values = []
#     while current in came_from:
#         path.append(current)
#         f_values.append(f_score[current])
#         current, action = came_from[current]
#         actions.append(action)
#     path.append(current)  # Add the start node
#     f_values.append(f_score[current])
#     return path[::-1], actions[::-1], f_values[::-1]

# def format_output(start, goal, grid, path, actions, f_values, nodes_generated):
#     output = f"{len(actions)}\n"
#     output += f"{nodes_generated}\n"
#     output += ' '.join(map(str, actions)) + "\n"
#     output += ' '.join(f"{f:.1f}" for f in f_values) + "\n"

#     for y, row in enumerate(grid):
#         for x, cell in enumerate(row):
#             if (y, x) == start:
#                 output += "2 "
#             elif (y, x) == goal:
#                 output += "5 "
#             elif (y, x) in path[1:-1]:
#                 output += "4 "
#             else:
#                 output += f"{cell} "
#         output += "\n"

#     return output.strip()

# def main(input_file, output_file, k):
#     start, goal, grid = read_input(input_file)
#     print(f"Start: {start}, Goal: {goal}")
#     print(f"Grid size: {len(grid)}x{len(grid[0])}")
    
#     result = astar(start, goal, grid, k)
    
#     if result[0] is not None:
#         path, actions, f_values, nodes_generated = result
#         output = format_output(start, goal, grid, path, actions, f_values, nodes_generated)
#         with open(output_file, 'w') as f:
#             f.write(output)
#         print(f"Path found! Check {output_file} for results.")
#         print(f"Path length: {len(path)}")
#         print(f"Nodes generated: {nodes_generated}")
#     else:
#         print("No path found.")
        
# if __name__ == "__main__":
#     input_file = "/Users/hariharanjanardhanan/Desktop/__/AI/2__.Proj1/AI-PROJECT1/Input1.txt"
#     output_file = "output.txt"
#     k = 2  # You can change this to 4 for the second run
#     main(input_file, output_file, k)


# Trail 2

import heapq
import math

def read_input(file_path):
    with open(file_path, 'r') as f:
        # Read start and goal coordinates (i,j format where i is column and j is row)
        start_i, start_j, goal_i, goal_j = map(int, f.readline().split())
        
        # Read the grid (top to bottom)
        grid_top_down = [list(map(int, line.split())) for line in f]
        
        # Reverse the grid to make row 0 at the bottom
        grid = grid_top_down[::-1]
        
        # Return coordinates in (i,j) format
        return (start_i, start_j), (goal_i, goal_j), grid
def format_output(start, goal, grid, path, actions, f_values, nodes_generated):
    # First line: depth (number of moves)
    output = f"{len(actions)}\n"
    
    # Second line: total nodes generated
    output += f"{nodes_generated}\n"
    
    # Third line: sequence of moves
    output += ' '.join(map(str, actions)) + "\n"
    
    # Fourth line: f(n) values
    output += ' '.join(f"{f:.1f}" for f in f_values) + "\n"
    
    # Lines 5-34: Modified grid (print from top to bottom)
    # Note: We need to print the grid in reverse order since (0,0) is at the bottom-left
    for j in range(len(grid)-1, -1, -1):  # Start from top row (29) down to bottom row (0)
        for i in range(len(grid[0])):      # Left to right (0 to 49)
            if (i, j) == start:
                output += "2 "
            elif (i, j) == goal:
                output += "5 "
            elif (i, j) in path[1:-1]:  # Path excluding start and goal
                output += "4 "
            else:
                output += f"{grid[j][i]} "
        output += "\n"
    
    return output.strip()
def heuristic(a, b):
    dx = abs(a[1] - b[1])
    dy = abs(a[0] - b[0])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def get_neighbors(node, grid):
    # Eight possible moves in (di,dj) format
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1),
                 (-1, 0), (-1, -1), (0, -1), (1, -1)]
    neighbors = []
    
    for idx, (di, dj) in enumerate(directions):
        ni = node[0] + di  # new column
        nj = node[1] + dj  # new row
        # Check bounds: i (0-49), j (0-29)
        if (0 <= ni < len(grid[0]) and 0 <= nj < len(grid) and 
            grid[nj][ni] != 1):  # Note: grid[j][i] since j is row, i is column
            neighbors.append((idx, (ni, nj)))
    return neighbors

# def get_neighbors(node, grid):
#     # Eight possible moves (i,j) format where i is column and j is row
#     directions = [(1, 0), (1, 1), (0, 1), (-1, 1),
#                  (-1, 0), (-1, -1), (0, -1), (1, -1)]
#     neighbors = []
    
#     current_i, current_j = node
#     for move_index, (di, dj) in enumerate(directions):
#         new_i = current_i + di
#         new_j = current_j + dj
        
#         # Check bounds: i (0-49), j (0-29)
#         if (0 <= new_i < 50 and 0 <= new_j < 30 and 
#             grid[new_j][new_i] != 1):  # Note: grid[j][i] since j is row, i is column
#             neighbors.append((move_index, (new_i, new_j)))
    
#     return neighbors
def angle_cost(prev_move=None,next_move=None,k=2):
     if prev_move==None:
         return(0)
     diff=abs(next_move-prev_move)
     if diff>4:
         diff=8-diff
     return(k*(diff/8))

def distance_cost(move):
     if move%2==0:
         return(1)
     else:
         return(math.sqrt(2))
def reconstruct_path(came_from, current, g_score, f_score):
    path = []
    actions = []
    f_values = []
    while current in came_from:
        path.append(current)
        f_values.append(f_score[current])
        current, action = came_from[current]
        actions.append(action)
    path.append(current)  # Add the start node
    f_values.append(f_score[current])
    return path[::-1], actions[::-1], f_values[::-1]

def astar(start, goal, grid, k, max_iterations=100000):
    open_list = [(0, start, None)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()
    nodes_generated = 1
    iterations = 0

    # Corrected loop condition
    while open_list and iterations < max_iterations:
        iterations += 1
        current_f, current, prev_move = heapq.heappop(open_list)
        
        if current == goal:
            print("Goal reached!")
            path, actions, f_values = reconstruct_path(came_from, current, g_score, f_score)
            return path, actions, f_values, nodes_generated

        closed_set.add(current)

        for move_index, neighbor in get_neighbors(current, grid):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + distance_cost(move_index) + angle_cost(prev_move, move_index, k)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = (current, move_index)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor, move_index))
                nodes_generated += 1

    print(f"No path found after exploring {nodes_generated} nodes.")
    return None
def main(input_file, output_file, k):
    start, goal, grid = read_input(input_file)
    print(f"Start: {start}, Goal: {goal}")
    print(f"Grid size: {len(grid)}x{len(grid[0])}")
    
    result = astar(start, goal, grid, k)
    
    if result is not None:
        path, actions, f_values, nodes_generated = result
        output = format_output(start, goal, grid, path, actions, f_values, nodes_generated)
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"Path found! Check {output_file} for results.")
        print(f"Path length: {len(path)}")
        print(f"Nodes generated: {nodes_generated}")
    else:
        print("No path found.")
        with open(output_file, 'w') as f:
            f.write("No path found.\n")
        
if __name__ == "__main__":
    input_file = "/Users/hariharanjanardhanan/Desktop/__/AI/2__.Proj1/AI-PROJECT1/Sample input.txt"
    output_file = "output.txt"
    k = 2  # You can change this to 4 for the second run
    main(input_file, output_file, k)