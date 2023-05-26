## <div align="center">2023-1 서강 융합기술 경진대회 </div>
🚀 yolov5 모델을 활용한 재활용품 분류 SW 및 장치

## <div align="center">Team 서계</div>
🌟 Team Leader 이창재 (서강대학교 기계공학과 19)

🌟 Team member 강정훈 (서강대학교 기계공학과 19)

🌟 Team member 김기훈 (서강대학교 기계공학과 / 컴퓨터공학과 19)

🌟 Team member 이도헌 (서강대학교 기계공학과 19)

## <div align="center">Summary</div>
🚀 Customized yolov5 using Roboflow and Google Colab

- [Our Roboflow Data Set](https://app.roboflow.com/sgme/classify-pet-and-can/4)

- yolov5m 모델을 이용하여 best.pt를 제작


🚀 Take Image using WebCam - terminal (fswebcam)


```python
import os
picture = "fswebcam --no-banner --set brightness=60% Images/test1.jpg"
os.system(picture)
```


🚀 Analysis the Image - customized yolov5 (detect.py and best.pt)

- Final Image

<p align="center"><img width="800" src="https://file.notion.so/f/s/f78846c6-a9a6-427b-a0f3-4428d04d011c/Untitled.jpeg?id=2e4f2ae3-20c1-45e3-b12a-bc9c71667c68&table=block&spaceId=89f4f652-5ebd-4c52-8a2d-be12a0e49dda&expirationTimestamp=1685187034925&signature=IfXaNab3w6zn-yrGZZgv5DQYqLhXkGZpFzqQxoTMwFQ&downloadName=Untitled.jpeg"></p>
 

- Object 좌표값 데이터를 output.txt에 저장

<p align="center">detect.py</p>

```python
if len(det):
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
    #Customized part : Save det(coordinate info) to output.txt
    for i in range(len(det)):
        print(float(det[i][0]))
        print(float(det[i][1]))
        print(float(det[i][2]))
        print(float(det[i][3]))
        print(float(det[i][5]))
```

<p align="center">Terminal</p>

```bash
# Modify file location
python3 yolov5/detect.py > yolov5/output.txt --weights yolov5/best.pt --img 640 --conf 0.4 --source Images/test1.jpg
```
    
🚀 Make Object Map using 2D Matrix from output.txt Data - Python3 Code(file_read.py) 

<p align="center">file_read.py</p>

```python
#Make 2D Array using output.txt (object location data)
def map():
        f = open("/home/sgme/yolov5/output.txt", 'r') # Modify file location
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
        
        # PET Area = ('p', k) k is object number
        # CAN Area = ('c', k) k is object number
        for m in range(0, rows-4):
            y_m = m*(720/(rows-4)) + (720/(2*(rows-4)))
            for n in range(0, cols):
                x_m = n*(1020/cols) + (1020/(2*cols))
                for k in range(len(object_list)):
                    if (x_m > object_list[k][0]) and (x_m < object_list[k][2]) and (y_m > object_list[k][1]) and (y_m < object_list[k][3]):
                        if object_list[k][4] == 1:
                            result_map[m+2][n] = ('p',k+1)
                            break
                        elif object_list[k][4] == 0:
                            result_map[m+2][n] = ('c',k+1)
                            break
        return result_map    
```

🚀 Find the Most Efficient Way - Python3 Code (dijkstra.py)

<details open>
<summary>dijkstra.py</summary>

```python
import heapq
import numpy as np
global INF
INF = 2550

def object_boundary(maze, row_index, col_index):
    m = len(maze) #Row of maze
    n = len(maze[0]) #Column of maze
    #valid index
    for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
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
    for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
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
```
 
</detail>
 

🚀 Send the Control Order to Arduino Nano by Serial Module - Python3 pyserial
 
```python
import serial

#Serial Setup
py_serial = serial.Serial(port = '/dev/ttyUSB0', baudrate = 9600)

#Get move_order from dijkstra	
	
while len(move_order):
	py_serial.write(move_order.pop(0))
	print("pop")
	time.sleep(1.0) #delay time decided by velocity and distance
```		
 
 
🚀 Recieve the Control Order in the Arduino - Arduino Serial (*.ino)

	
<p align = "center">motor_control.ino</p>
```cpp
// pin
#define X_dirPin 2
#define X_stepPin 3
#define Y_dirPin 4
#define Y_stepPin 5
#define Z_dirPin 6
#define Z_stepPin 7


// const
float sPR = 3200;
int Head_dir = 0;

void setup() {
  Serial.begin(9600);
  pinMode(X_dirPin,OUTPUT);
  pinMode(Y_dirPin,OUTPUT);
  pinMode(X_stepPin,OUTPUT);
  pinMode(Y_stepPin,OUTPUT);
  pinMode(Z_dirPin,OUTPUT);
  pinMode(Z_stepPin,OUTPUT);
}

void move(int x_distance, int y_distance, int Head){

  float dx, dy;
  dx = (float)x_distance/80; //원래 100
  dy = (float)y_distance/80;

  int Head_new = Head;
  
  int angle = Head_new - Head_dir;
    

  if(angle !=0){
    if(abs(angle)<4){
      if(angle>0){
          digitalWrite(Z_dirPin, HIGH);
      }
      else{
          digitalWrite(Z_dirPin, LOW);
      }
      Head_dir = Head_new;
      for(int i=0; i<sPR*abs(angle)/8; i++){
        digitalWrite(Z_stepPin,HIGH);
        delayMicroseconds(200);
        digitalWrite(Z_stepPin,LOW);
        delayMicroseconds(200);
      }
    }
    else{
      if(angle>0){
        digitalWrite(Z_dirPin,LOW); 
      }
      else{
        digitalWrite(Z_dirPin,HIGH);
      }
        
      Head_dir = Head_new;

      for(int i=0; i<sPR*(8-abs(angle))/8; i++){
        digitalWrite(Z_stepPin,HIGH);
        delayMicroseconds(200);
        digitalWrite(Z_stepPin,LOW);
        delayMicroseconds(200);
      }    
    }
  }
   

    
    switch(Head_new){
      case 0:
        digitalWrite(X_dirPin, HIGH);
        digitalWrite(Y_dirPin, LOW);
        break;
      case 1:
        digitalWrite(X_dirPin, HIGH);
        digitalWrite(Y_dirPin, LOW);
        break;
      case 2:
        digitalWrite(X_dirPin, HIGH);
        digitalWrite(Y_dirPin, LOW);
        break;
      case 3:
        digitalWrite(X_dirPin, HIGH);
        digitalWrite(Y_dirPin, HIGH);
        break;
      case 4:
        digitalWrite(X_dirPin, LOW);
        digitalWrite(Y_dirPin, HIGH);
        break;
      case 5:
        digitalWrite(X_dirPin, LOW);
        digitalWrite(Y_dirPin, HIGH);
        break;
      case 6:
        digitalWrite(X_dirPin, LOW);
        digitalWrite(Y_dirPin, HIGH);
        break;
      case 7:
        digitalWrite(X_dirPin, LOW);
        digitalWrite(Y_dirPin, LOW);
        break;
        
    }
  
    // X,Y 거리, 속도
    for(int i=0, j=0; (i<100000*dx)||(j<100000*dy);i++,j++){
      if(i<100000*dx){
        digitalWrite(X_stepPin,HIGH);
        delayMicroseconds(30);
        digitalWrite(X_stepPin,LOW);
        delayMicroseconds(30);
      }
      if((j<100000*dy)){
        digitalWrite(Y_stepPin,HIGH);
        delayMicroseconds(30);
        digitalWrite(Y_stepPin,LOW);
        delayMicroseconds(30);
      }
    }
}



void loop() {
  int data[3];
  if(Serial.available() >= 3){
    for (int i = 0; i < 3; i++){
      data[i] = Serial.read();
    }
    int x_d = data[1];
    int y_d = data[0];
    int z_dir = data[2];
    move(x_d, y_d, z_dir);
  }
  Serial.println("done");
}
```
	
🚀 Integration Code - Python3 Code (recycle.py)

<p align = "center"> recycle.py </p>

```python

	
```
			  
			  

## <div align="center">Usecase Diagram</div>



## <div align="center">System Context Class Diagram</div>



## <div align="center">Object Structuring</div>

## <div align="center">Collaboration Diagram</div>

## <div align="center">Documentation</div>

## <div align="center">Reference - yolov5</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com/yolov5) for full documentation on training, testing and deployment. See below for quickstart examples.

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>

<details>
<summary>Inference</summary>

YOLOv5 [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading) inference. [Models](https://github.com/ultralytics/yolov5/tree/master/models) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --weights yolov5s.pt --source 0                               # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
results. [Models](https://github.com/ultralytics/yolov5/tree/master/models)
and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training times for YOLOv5n/s/m/l/x are
1/2/4/6/8 days on a V100 GPU ([Multi-GPU](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training) times faster). Use the
largest `--batch-size` possible, or pass `--batch-size -1` for
YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092). Batch sizes shown for V100-16GB.

```bash
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>

<details open>
<summary>Tutorials</summary>

- [Train Custom Data](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data) 🚀 RECOMMENDED
- [Tips for Best Training Results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results) ☘️
- [Multi-GPU Training](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training)
- [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading) 🌟 NEW
- [TFLite, ONNX, CoreML, TensorRT Export](https://docs.ultralytics.com/yolov5/tutorials/model_export) 🚀
- [NVIDIA Jetson platform Deployment](https://docs.ultralytics.com/yolov5/tutorials/running_on_jetson_nano) 🌟 NEW
- [Test-Time Augmentation (TTA)](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation)
- [Model Ensembling](https://docs.ultralytics.com/yolov5/tutorials/model_ensembling)
- [Model Pruning/Sparsity](https://docs.ultralytics.com/yolov5/tutorials/model_pruning_and_sparsity)
- [Hyperparameter Evolution](https://docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution)
- [Transfer Learning with Frozen Layers](https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers)
- [Architecture Summary](https://docs.ultralytics.com/yolov5/tutorials/architecture_description) 🌟 NEW
- [Roboflow for Datasets, Labeling, and Active Learning](https://docs.ultralytics.com/yolov5/tutorials/roboflow_datasets_integration)
- [ClearML Logging](https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration) 🌟 NEW
- [YOLOv5 with Neural Magic's Deepsparse](https://docs.ultralytics.com/yolov5/tutorials/neural_magic_pruning_quantization) 🌟 NEW
- [Comet Logging](https://docs.ultralytics.com/yolov5/tutorials/comet_logging_integration) 🌟 NEW

</details>

## <div align="center">Why YOLOv5</div>

YOLOv5 has been designed to be super easy to get started and simple to learn. We prioritize real-world results.

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure</summary>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040757-ce0934a3-06a6-43dc-a979-2edbbd69ea0e.png"></p>
</details>
<details>
  <summary>Figure Notes</summary>

- **COCO AP val** denotes mAP@0.5:0.95 metric measured on the 5000-image [COCO val2017](http://cocodataset.org) dataset over various inference sizes from 256 to 1536.
- **GPU Speed** measures average inference time per image on [COCO val2017](http://cocodataset.org) dataset using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100 instance at batch-size 32.
- **EfficientDet** data from [google/automl](https://github.com/google/automl) at batch size 8.
- **Reproduce** by `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pt yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`

</details>

### Pretrained Checkpoints

| Model                                                                                           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
| ----------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------- | ---------------------------- | ----------------------------- | ------------------------------ | ------------------ | ---------------------- |
| [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)              | 640                   | 28.0                 | 45.7              | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)              | 640                   | 37.4                 | 56.8              | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   |
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)              | 640                   | 45.4                 | 64.1              | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt)              | 640                   | 49.0                 | 67.3              | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt)              | 640                   | 50.7                 | 68.9              | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  |
|                                                                                                 |                       |                      |                   |                              |                               |                                |                    |                        |
| [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt)            | 1280                  | 36.0                 | 54.4              | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |
| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt)            | 1280                  | 44.8                 | 63.7              | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |
| [YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt)            | 1280                  | 51.3                 | 69.3              | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt)            | 1280                  | 53.7                 | 71.3              | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |
| [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt)<br>+ [TTA] | 1280<br>1536          | 55.0<br>**55.8**     | 72.7<br>**72.7**  | 3136<br>-                    | 26.2<br>-                     | 19.4<br>-                      | 140.7<br>-         | 209.8<br>-             |

<details>
  <summary>Table Notes</summary>

- All checkpoints are trained to 300 epochs with default settings. Nano and Small models use [hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml) hyps, all others use [hyp.scratch-high.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-high.yaml).
- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.<br>Reproduce by `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
- **Speed** averaged over COCO val images using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance. NMS times (~1 ms/img) not included.<br>Reproduce by `python val.py --data coco.yaml --img 640 --task speed --batch 1`
- **TTA** [Test Time Augmentation](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation) includes reflection and scale augmentations.<br>Reproduce by `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>


## <div align="center">Contribute</div>

We love your input! We want to make contributing to YOLOv5 as easy and transparent as possible. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) to get started, and fill out the [YOLOv5 Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) to send us feedback on your experiences. Thank you to all our contributors!

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" /></a>

## <div align="center">License</div>

YOLOv5 is available under two different licenses:

- **AGPL-3.0 License**: See [LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) file for details.
- **Enterprise License**: Provides greater flexibility for commercial product development without the open-source requirements of AGPL-3.0. Typical use cases are embedding Ultralytics software and AI models in commercial products and applications. Request an Enterprise License at [Ultralytics Licensing](https://ultralytics.com/license).
  
  
  
  

