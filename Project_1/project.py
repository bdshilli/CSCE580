import time
import numpy as np
from gridgame import *

##############################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.
# Please do not modify or remove lines 18 and 19.

##############################################################################################################################

game = ShapePlacementGrid(GUI=True, render_delay_sec=0., gs=7, num_colored_boxes=6)
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')
np.savetxt('initial_grid.txt', grid, fmt="%d")

##############################################################################################################################

# Initialization

# shapePos is the current position of the brush.

# currentShapeIndex is the index of the current brush type being placed (order specified in gridgame.py, and assignment instructions).

# currentColorIndex is the index of the current color being placed (order specified in gridgame.py, and assignment instructions).

# grid represents the current state of the board. 
    
    # -1 indicates an empty cell
    # 0 indicates a cell colored in the first color (indigo by default)
    # 1 indicates a cell colored in the second color (taupe by default)
    # 2 indicates a cell colored in the third color (veridian by default)
    # 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.
    
    # Each shape is represented as a list containing three elements: a) the brush type (number between 0-8), 
    # b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between 0-3)

    # For instance [0, (0,0), 2] represents a shape spanning a single cell in the color 2=veridian, placed at the top left cell in the grid.

# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

##############################################################################################################################

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

# input()   # <-- workaround to prevent PyGame window from closing after execute() is called, for when GUI set to True. Uncomment to enable.
#print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)


####################################################
# Timing your code's execution for the leaderboard.
####################################################

start = time.time()  # <- do not modify this.



##########################################
# Write all your code in the area below. 
##########################################

used_colors = set()
used_shapes = set()

def getObjectiveValue(grid, placedShapes):
    empty_cells = np.sum(grid == -1)
    temp_used_shapes = set()
    temp_used_colors = set()
    for shape_index, _, color_index in placedShapes:
        temp_used_shapes.add(shape_index)
        temp_used_colors.add(color_index)
    # Minimize the number of empty cells, colors and shapes used
    return 2*empty_cells + (0.1)*len(temp_used_colors) + (0.1)*len(temp_used_shapes)

def check_valid_placement(grid, shape, pos, color):
    # gridSize = len(grid)
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                if pos[0] + j >= game.gridSize or pos[1] + i >= game.gridSize:
                    return False
                if grid[pos[1] + i, pos[0] + j] != -1:
                    return False
                if pos[1] + i > 0 and grid[pos[1] + i-1][pos[0] + j] == color:
                    return False
                if pos[1] + i < len(grid) - 1 and grid[pos[1] + i+1][pos[0] + j] == color:
                    return False
                if pos[0] + j > 0 and grid[pos[1] + i][pos[0] + j-1] == color:
                    return False
                if pos[0] + j < len(grid) - 1 and grid[pos[1] + i][pos[0] + j+1] == color:
                    return False

    return True

def get_neighbor_states(game, placedShapes):
    neighbors = []
    shapePos, currentShapeIndex, _, grid, placedShapes, done = game.execute('export')

    while currentShapeIndex != 0:
        game.execute('switchshape')
        _, currentShapeIndex, _, _, _, _ = game.execute('export')
    if grid[shapePos[1], shapePos[0]] != -1:
        i, j = np.argwhere(grid == -1)[0]
        while done or shapePos != [j,i]:
            #print("try", shapePos, [j,i])
            if shapePos[0] < j:
                game.execute('right')
            elif shapePos[0] > j:
                game.execute('left')
            if shapePos[1] < i:
                game.execute('down')
            elif shapePos[1] > i:
                game.execute('up')
            shapePos, _, _, _, _, _ = game.execute('export')
        #print("ShapePos New", shapePos)
    
    for shape_index in range(len(game.shapes)):
        _, currentShapeIndex, currentColorIndex, _, _, _ = game.execute('export')
        while shape_index != currentShapeIndex:
            game.execute('switchshape')
            shapePos, currentShapeIndex, _, _, _, _ = game.execute('export')
        for color_index in range(4):
            if check_valid_placement(grid, game.shapes[currentShapeIndex], shapePos, color_index):
                #print(currentShapeIndex, shapePos, color_index)
                while color_index != currentColorIndex:
                    game.execute('switchcolor')
                    _, _, currentColorIndex, _, _, _ = game.execute('export')
                shapePos_new, currentShapeIndex_new, currentColorIndex_new, grid_new, placedShapes_new, done_new = game.execute('place')
                #print("Place", placedShapes_new)
                # neighbors.append((grid_new.copy(), placedShapes_new.copy()))
                neighbors.append((shapePos_new.copy(), currentShapeIndex_new, currentColorIndex_new, grid_new.copy(), placedShapes_new.copy(), done_new))

                #print("Neighbor", neighbors)
                game.execute('undo')
                #print("Undo")
                #print("Neighbor", neighbors)
    
    return neighbors

def hill_climbing(game, grid, placedShapes):
    # current_grid = grid.copy()
    # current_placedShapes = placedShapes.copy()
    _, _, _, current_grid, current_placedShapes, done = game.execute('export') 

    while True:
        # Generate neighboring states
        #print("Current grid", current_grid)
        neighbors = get_neighbor_states(game, current_placedShapes)
        #print("Neighbors", len(neighbors))
        
        # Find the best neighbor
        best_neighbor = None
        best_score = getObjectiveValue(current_grid, current_placedShapes)
        #print("Curr score:", best_score)
        if game.checkGrid(current_grid):
            #print("Correct grid")
            break
        temp_placedShapes = current_placedShapes.copy()
        for neighbor_shapePos, neighbor_shapeInd, neighbor_colorInd, neighbor_grid, neighbor_placedShapes, neighbor_done in neighbors:
            #print("Neighbor...", neighbor_placedShapes)
            neighbor_score = getObjectiveValue(neighbor_grid, neighbor_placedShapes)
            #print("Neighbor score:", neighbor_score)
            if neighbor_score < best_score or (neighbor_score==best_score and len(neighbor_placedShapes)<len(temp_placedShapes)) or (neighbor_score == best_score and neighbor_colorInd in used_colors) or (neighbor_score == best_score and neighbor_shapeInd in used_shapes):
                best_neighbor = (neighbor_shapePos, neighbor_shapeInd, neighbor_colorInd, neighbor_grid, neighbor_placedShapes, neighbor_done)
                best_score = neighbor_score
                temp_placedShapes = neighbor_placedShapes.copy()
            
        #print("Best neighbor:", best_neighbor)
        # If no better neighbor is found, stop

        if best_neighbor is None:
            break

        shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')
        while shapePos != best_neighbor[0]:
            if shapePos[0] < best_neighbor[0][0]:
                game.execute('right')
            elif shapePos[0] > best_neighbor[0][0]:
                game.execute('left')
            if shapePos[1] < best_neighbor[0][1]:
                game.execute('down')
            elif shapePos[1] > best_neighbor[0][1]:
                game.execute('up')
            shapePos, _, _, _, _, _ = game.execute('export')
        while currentShapeIndex != best_neighbor[1]:
            game.execute('switchshape')
            _, currentShapeIndex, _, _, _, _ = game.execute('export')
        while currentColorIndex != best_neighbor[2]:
            game.execute('switchcolor')
            _, _, currentColorIndex, _, _, _ = game.execute('export')
        game.execute('place')
        _, _, _, current_grid, current_placedShapes, done = game.execute('export')

        if done:
            break
    
    return current_grid, current_placedShapes

def calculateUsedColors(game, grid):
    """
    Calculates the number of colors used in the grid.
    """
    for i in range(game.gridSize):
        for j in range(game.gridSize):
            if grid[i, j] != -1:
                used_colors.add(grid[i, j])
    return 


calculateUsedColors(game, grid)
# Step 1: Initialize the grid to a valid starting state
# grid, placedShapes = initialize_valid_grid(game)

# Step 2: Run hill climbing on the valid grid
grid, placedShapes = hill_climbing(game, grid, placedShapes)


# input()
########################################

# Do not modify any of the code below. 

########################################

end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))