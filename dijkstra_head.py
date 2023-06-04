import heapq
import numpy as np
global INF
INF = 2550

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

#Function : Make head position index list from given map
def find_head_index(given_map):
    row_total = 0
    col_total = 0
    for i in range(len(given_map)):
        for j in range(len(given_map[0])):
            if given_map[i][j][0] == 'h':
                row_total += i
                col_total += j
    start_row = round(row_total)
    start_col = round(col_total)
    return start_row, start_col

#Main
def main(given_map):
    start_row, start_col = find_head_index(given_map)
    
    start = (start_row, start_col)
    print("START AT :", start, "\n")

    end_row = 0
    end_col = 0
    end = (end_row, end_col)
    
    path = find_shortest_path(given_map, start, end)
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
        
    print("RESULT MOVE ORDER : \n", move_order, "\n")
 
    return move_order
