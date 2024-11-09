# Authors - Tewoflos Girmay and Hariharan Janardhanan
import heapq
import math
import matplotlib.pyplot as plt
import numpy as np


def read_input(file_path):
    with open(file_path, 'r') as f:
        # Read start and goal coordinates
        start_i, start_j, goal_i, goal_j = map(int, f.readline().split())
        
        # Initialize grid
        grid = []
        
        # Read each line and split it into a list of integers
        for line in f:
            # Split the line by spaces and convert to integers
            row = [int(x) for x in line.strip().split()]
            if len(row) == 50:  # Ensure each row has 50 columns
                grid.append(row)
        
        # Verify grid dimensions
        if len(grid) != 30 or len(grid[0]) != 50:
            print(f"Current dimensions: {len(grid)}x{len(grid[0])}")
            raise ValueError(f"Invalid grid dimensions: {len(grid)}x{len(grid[0])}. Expected 30x50")
        
        # Reverse the grid to make row 0 at the bottom
        grid.reverse()
        
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
    
    # To print the grid in reverse order since (0,0) is at the bottom-left
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
    # Euclidean distance
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def get_neighbors(node, grid):
    # Eight possible moves in clockwise order starting from East (0)
    directions = [(1, 0),   # 0: East (right)
                 (1, 1),    # 1: Northeast (right & up)
                 (0, 1),    # 2: North (up)
                 (-1, 1),   # 3: Northwest (left & up)
                 (-1, 0),   # 4: West (left)
                 (-1, -1),  # 5: Southwest (left & down)
                 (0, -1),   # 6: South (down)
                 (1, -1)]   # 7: Southeast (right & down)
    neighbors = []
    for idx, (di, dj) in enumerate(directions):
        ni = node[0] + di  # new column
        nj = node[1] + dj  # new row
        # Check bounds: i (0-49), j (0-29)
        if (0 <= ni < len(grid[0]) and 0 <= nj < len(grid) and 
            grid[nj][ni] != 1):  # grid[j][i] since j is row, i is column
            neighbors.append((idx, (ni, nj)))
    return neighbors


def angle_cost(prev_move=None,next_move=None,k=2):
     if prev_move==None:# No angle cost for the first move from start position
         return(0)
     diff=abs(next_move-prev_move) # Calculate absolute difference between movement directions
     if diff>4:   # If difference is more than 180 degrees (4 moves),
         diff=8-diff
     return(k*(diff/8)) #Calculate final cost: k * (angle_difference / total_possible_directions)


def distance_cost(move):
     if move%2==0:     # Even-numbered moves (0,2,4,6) are horizontal/vertical
         return(1)
     else:     # Odd-numbered moves (1,3,5,7) are diagonal
         return(math.sqrt(2))

def reconstruct_path(came_from, current, g_score, f_score):
    path = []
    actions = []
    f_values = [] # Will store the f-scores of nodes along the path
    while current in came_from:
        path.append(current) # Add the current node to the beginning of the path
        f_values.append(f_score[current])
        current, action = came_from[current] # Get the parent of the current node and the action that led to the current node
        actions.append(action)
    path.append(current)  # Add the start node
    f_values.append(f_score[current])
    return path[::-1], actions[::-1], f_values[::-1]     # Reverse lists to get path from start to goal


def astar(start, goal, grid, k, max_iterations=100000):
    # Initialize open list with start node (f_score, node, previous_move)

    open_list = [(0, start, None)]
    came_from = {}  # Dictionary to store path information
    g_score = {start: 0}  # Cost from start to current nod
    f_score = {start: heuristic(start, goal)} # Total estimated cost
    closed_set = set() # Set of explored nodes
    nodes_generated = 1 # Set of explored nodes
    iterations = 0

    while open_list and iterations < max_iterations:
        iterations += 1
        # Get node with lowest f_score from open list
        current_f, current, prev_move = heapq.heappop(open_list)
        
        if current == goal:         # Check if goal is reached
            print("Goal reached!")
            path, actions, f_values = reconstruct_path(came_from, current, g_score, f_score)
            return path, actions, f_values, nodes_generated
        # Add current node to explored set
        closed_set.add(current)

        for move_index, neighbor in get_neighbors(current, grid): # Explore neighbors of current node
            if neighbor in closed_set:
                continue
            # Calculate new path cost including angle and distance costs
            tentative_g_score = g_score[current] + distance_cost(move_index) + angle_cost(prev_move, move_index, k)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:             # If new path is better, update the node
                came_from[neighbor] = (current, move_index)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor, move_index))
                nodes_generated += 1

    print(f"No path found after exploring {nodes_generated} nodes.")
    return None

def visualize_grid(grid, path, start, goal):
    # Create an empty grid for visualization
    visual_grid = np.zeros_like(grid, dtype=int)

    # Mark the grid with corresponding colors
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:  # Illegal state (obstacle)
                visual_grid[i][j] = 1  # Black
            elif (j, i) == start:  # Start state
                visual_grid[i][j] = 2  # Yellow
            elif (j, i) == goal:  # Goal state
                visual_grid[i][j] = 3  # Green
            elif (j, i) in path:  # Path state
                visual_grid[i][j] = 4  # White for legal states in the path

    # Define a custom color map for the grid
    cmap = plt.cm.colors.ListedColormap(['white','black','yellow', 'green'])
    bounds = [0, 1, 2, 3, 4]
    norm = plt.Normalize(vmin=0, vmax=4)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(visual_grid, cmap=cmap, origin='lower', extent=[-0.5, len(grid[0]) - 0.5, -0.5, len(grid) - 0.5], norm=norm)

    # Mark start and goal on the grid
    start_x, start_y = start
    goal_x, goal_y = goal
    ax.scatter(start_x, start_y, marker='o', color='red', s=200, label="Start")
    ax.scatter(goal_x, goal_y, marker='X', color='yellow', s=200, label="Goal")

    # Adding grid lines for better visualization
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)

    # Adding legend
    ax.legend()
    ax.set_title("A* Pathfinding Visualization")
    plt.show()

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
        
        # Visualize the grid with the path, start, and goal
        visualize_grid(grid, path, start, goal)
    else:
        print("No path found.")
        with open(output_file, 'w') as f:
            f.write("No path found.\n")
        
if __name__ == "__main__":
    input_file = 'Input.txt'
    output_file = 'output_file.txt'  # Replace with your output file path
    k = 2  # You can adjust the penalty factor 'k' for angle change cost
    main(input_file, output_file, k)



     
