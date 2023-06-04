import heapq
import numpy as np
global INF
INF = 2550

def object_boundary(maze, row_index, col_index):
    m = len(maze) #Row of maze
    n = len(maze[0]) #Column of maze
    #valid index
    for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2), (-2, -1)]:
            next_y, next_x = row_index + dy, col_index + dx
            # Case : Index out of range
            if next_y < 0 or next_y >= m or next_x < 0 or next_x >= n:
                continue
            else :
              maze[next_y][next_x] = -1

def object_boundary_deletion(maze, row_index, col_index):
    m = len(maze) #Row of maze
    n = len(maze[0]) #Column of maze
    #valid index
    for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2), (-2, -1)]:
            next_y, next_x = row_index + dy, col_index + dx
            # Case : Index out of range
            if next_y < 0 or next_y >= m or next_x < 0 or next_x >= n:
                continue
            else :
              maze[next_y][next_x] = 0

#Function : Dijkstra - Calculate distance of all point from start point
def dijkstra(maze, start):
    m = len(maze) #Row of maze
    n = len(maze[0]) #Column of maze
    dist = np.array([[INF] * n for _ in range(m)]) #Fill the maze with INF
    dist[start[0]][start[1]] = 0 #Set the start point as 0
    heap = [(0, start)] #Initial heap which has start point and initial cost
    
    #Until heap element exists
    while heap:
        cost, pos = heapq.heappop(heap) #Pop the element from heap
        # Case : The smaller value of cost already exists
        if dist[pos[0]][pos[1]] < cost:
            continue
        
        #N, NE, E, SE, S, SW, W, NW -> 8 direction Searching from current position
        for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
            #Next position
            next_y, next_x = pos[0] + dy, pos[1] + dx
            # Case : Index out of range
            if next_y < 0 or next_y >= m or next_x < 0 or next_x >= n:
                continue
            # Case : Obstacle
            if maze[next_y][next_x] == -1:
                continue
            # Cost of next position
            if (dy == 0 or dx == 0):
                next_cost = cost + 10 
            else : 
                next_cost = cost + 14
            # Case : Update cost of next position to minimum cost
            if next_cost < dist[next_y][next_x]:
                dist[next_y][next_x] = next_cost
                #Update heap
                heapq.heappush(heap, (next_cost, (next_y, next_x)))
    return dist

#Function : Find shortest path from start point to target point
def find_shortest_path(maze, start, target):
    object_boundary_deletion(maze, start[0], start[1])
    object_boundary_deletion(maze, target[0], target[1])
    dist = dijkstra(maze, start)
    m = len(maze) #Row of maze
    n = len(maze[0]) #Column of maze
    path = [(target[0], target[1])] #stack of path
    #Until start point
    while path[-1] != start:
        y, x = path[-1]
        min_dist = dist[y][x] #Current point
        for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
            next_y, next_x = y + dy, x + dx
            # Case : Index out of range
            if next_y < 0 or next_y >= m or next_x < 0 or next_x >= n:
                continue
            # Case : Current point > Precede point -> update
            if dist[next_y][next_x] < min_dist:
                min_dist = dist[next_y][next_x]
                next_pos = (next_y, next_x)
        path.append(next_pos)
    object_boundary(maze, start[0], start[1])
    object_boundary(maze, target[0], target[1])
    return path

#Function : Make can and PET position index list from given map
def find_pet_can_list(given_map):
    pet_list = []
    can_list = []
    for i in range(len(given_map)):
        for j in range(len(given_map[0])):
            if given_map[i][j][0] == 'p':
                pet_list.append((i, j))
            elif given_map[i][j][0] == 'c':
                can_list.append((i, j))
    return pet_list, can_list

#Function : Make map which has shortest path from PET and Can to trash area
def find_shortest_classfy_path(maze, final_map, pet_list, can_list, start):
    #PET
    m = len(maze)
    n = len(maze[0])
    for i, j in pet_list:
        temp_maze = maze
        object_boundary_deletion(temp_maze, i, j)
        dist = dijkstra(temp_maze, (i, j))
        min_pet = INF
        #Find minimum distance from PET to trash area
        for k in range(len(maze[0])):
            if dist[0][k] < min_pet:
                min_pet = dist[0][k]
        final_map[[i],[j]] = min_pet
        object_boundary(temp_maze, i, j)
    #CAN
    for i, j in can_list:
        temp_maze = maze
        object_boundary_deletion(temp_maze, i, j)
        dist = dijkstra(temp_maze, (i, j))
        min_can = INF
        #Find minimum distance from can to trash area
        for k in range(len(maze[0])):
            if dist[len(maze)-1][k] < min_can:
                min_can = dist[len(maze)-1][k]
        final_map[[i],[j]] = min_can
        object_boundary(temp_maze, i, j)
    #Approach
    for i, j in pet_list+can_list:
        #Refactoring needed for efficiency
        temp_maze = maze
        temp_maze[i][j] = 0
        object_boundary_deletion(temp_maze, i, j)
        dist = dijkstra(temp_maze, start)
        final_map[[i],[j]] += dist[[i],[j]] 
        object_boundary(temp_maze, i, j)
    return final_map

#Function : Find most efficient index from fin al map
def find_most_efficient_index(map):
    min = INF
    min_index = (0, 0)
    for i in range(len(map)):
        for j in range(len(map[0])):
            if map[i][j] != 0 and map[i][j] < min:
                min = map[i][j]
                min_index = (i, j)
    return min_index

#Function : Classify trash using efficient path
def classify_trash(given_map, start):
    m = len(given_map)
    n = len(given_map[0]) 
    maze = np.zeros((m,n))
    final_map = np.zeros((m,n))
    
    pet_list, can_list = find_pet_can_list(given_map)
    print("REMAIN PET INDEX : ", pet_list, "\n")
    print("REMAIN CAN INDEX : ",can_list, "\n")   
    for i, j in pet_list+can_list:
        maze[i][j] = -1
        object_boundary(maze, i, j)

    final_map = find_shortest_classfy_path(maze, final_map, pet_list, can_list, start)
    print("FINAL MAP: \n", final_map, "\n")

    target = find_most_efficient_index(final_map)

    for i, j in pet_list+can_list:
        maze[i][j] = -1

    object_boundary_deletion(maze, target[0], target[1])
    dist = dijkstra(maze, target)
    final_path = []

    if given_map[target[0]][target[1]][0] == 'p':
        min_pet = INF
        min_pet_col = 0
        for i in range(len(maze[0])):
            if dist[0][i] < min_pet:
                min_pet = dist[0][i]        
                min_pet_col = i
        final_path = find_shortest_path(maze, target, (0, min_pet_col))
    else :
        min_can = INF
        min_can_col = 0
        for i in range(len(maze)):
            if dist[len(maze)-1][i] < min_can:
                min_can = dist[len(maze)-1][i]        
                min_can_col = i
        final_path = find_shortest_path(maze, target, (len(maze)-1, min_can_col))
        
    temp_path = find_shortest_path(maze, start, target)[1:]
    final_path += temp_path
    start = final_path[0]   
    return start, final_path, target, pet_list, can_list
    
#Main
def main(given_map, start_row, start_col):
    start = (start_row, start_col)
    print("START AT :", start, "\n")
    start, path, target, pet_list, can_list = classify_trash(given_map, start)
    end_row = start[0]
    end_col = start[1]
    target_number = given_map[target[0]][target[1]][1]
    target_object = given_map[target[0]][target[1]][0]
    
    print(target_object, target_number)
    print("RESULT INDEX: ")
    print(path, "\n")

    move_order = [[0,0,0]]

    row_1, col_1 = path.pop()
    while(path):
        row_2, col_2 = path.pop()
        dy = row_2 - row_1
        dx = col_2 - col_1
        #N
        if dy < 0 and dx == 0:
            new = [2, 0, 0]
        #NE
        elif dy < 0 and dx > 0:
            new = [2, 2, 1]
        #E
        elif dy == 0 and dx > 0:
            new = [0, 2, 2]
        #SE
        elif dy > 0 and dx > 0:
            new = [2, 2, 3]
        #S
        elif dy > 0 and dx == 0:
            new = [2, 0, 4]
        #SW
        elif dy > 0 and dx < 0:
            new = [2, 2, 5]
        #W
        elif dy == 0 and dx < 0:
            new = [0, 2, 6]
        #NW
        else :
            new = [2, 2, 7]

        if (move_order):
            old = move_order[-1]    
            if (old[2] == new[2]) and (old[0] + new[0] < 16) and (old[1] + new[1] < 16):
                move_order[-1][0] += new[0]
                move_order[-1][1] += new[1]
            else:
                move_order.append(new)
        else :
            move_order.append(new)
        row_1, col_1 = row_2, col_2
            
              
    for i in range(len(move_order)):
        move_order[i] = tuple(move_order[i])

    for i in range(len(given_map)):
        for j in range(len(given_map[0])):
            if (given_map[i][j] == (target_object, target_number) and (target_object, target_number) != ('e',0)) :
                given_map[i][j] = ('e', 0)
                print("delete ", i, j)
    print("RESULT MOVE ORDER : \n", move_order, "\n")
 
    return move_order, given_map, end_row, end_col, pet_list, can_list
