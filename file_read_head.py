#file_read_head.py
#read data from file
#Make 2D Array using output.txt (object location data)
def map():
        f = open("/home/sgme/yolov5/output1.txt", 'r')
        f.readline()
        object_list = []
        lines = f.read().splitlines()
        temp_list=[]
        count = 0
        for i in lines:
            if count == 5:
                count = 0
                object_list.append(temp_list)
                temp_list = []
            temp_list.append(round(float(i)))
            count+=1
        object_list.append(temp_list) #last object
        f.close

        # Make result_map to find efficient route 
        # width * height = 1020px * 720px, row x column = 24 x 34 2D MATRIX
        # Append 2 more row to each top and bottom
        # The top and bottom 2 row area will be trash area
        rows = 28
        cols = 34
        
        # Empty Area = ('e', 0)
        result_map = [[('e',0) for j in range(cols)] for i in range(rows)]
        
        # Head Area = ('h', k) k is object number
        for m in range(0, rows-4):
            y_m = m*(720/(rows-4)) + (720/(2*(rows-4)))
            for n in range(0, cols):
                x_m = n*(1020/cols) + (1020/(2*cols))
                if (x_m > object_list[0][0]) and (x_m < object_list[0][2]) and (y_m > object_list[0][1]) and (y_m < object_list[0][3]):
                        result_map[m+2][n] = ('h', 1)
        return result_map
