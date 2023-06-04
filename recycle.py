import os
import time
import serial
import cv2 as cv
import file_read
import file_read_head
import dijkstra
import dijkstra_head

#Serial Setup
py_serial = serial.Serial(port = '/dev/ttyUSB8', baudrate = 9600)

#Camera Setup
picture = "fswebcam --no-banner --set brightness=60% Images/test1.jpg"
os.system(picture)
img = cv.imread("Images/test1.jpg", cv.IMREAD_COLOR)
resize_img = cv.resize(img, (1020,720), interpolation=cv.INTER_AREA)
cv.imwrite("Images/test1.jpg", resize_img)

#Image Analysis (PET & CAN)
yolo1 = "python3 /home/sgme/yolov5/detect.py > /home/sgme/yolov5/output.txt --weights /home/sgme/yolov5/best.pt --img 640 --conf 0.4 --source /home/sgme/Images/test1.jpg"
os.system(yolo1)
time.sleep(1)

#Image Analysis (Head)
yolo2 = "python3 /home/sgme/yolov5/detect.py > /home/sgme/yolov5/output1.txt --weights /home/sgme/yolov5/best1.pt --img 640 --conf 0.4 --source /home/sgme/Images/test1.jpg"
os.system(yolo2)
time.sleep(1)

#Head position
lines = open('/home/sgme/yolov5/output1.txt').readlines()
given_map_head = file_read_head.map()
move_order_head = dijkstra_head.main(given_map_head)
py_serial.write(move_order.pop(0));
time.sleep(1.0)

#object list
lines = open('/home/sgme/yolov5/output.txt').readlines()
given_map=file_read.map()
start_row = 0
start_col = 0

while True :
	#Make move order from start position
	move_order, given_map, start_row, start_col, pet_list, can_list = dijkstra.main(given_map, start_row, start_col)
	#msg = "go"
	#py_serial.write(msg.encode())
	#time.sleep(2)
	if (len(pet_list) == 0 and len(can_list) == 0):
		print("FINISH\n")
		break
	
	while len(move_order):
		py_serial.write(move_order.pop(0))
		print("pop")
		time.sleep(1.0) #delay time decided by velocity and distance
		#return_msg = py_serial.readline().decode('utf-8')
		#print(return_msg)
		#while(return_msg[0:4] != "done"):
			#return_msg = py_serial.readline().decode('utf-8')
			#print(return_msg)
			#time.sleep(0.5)
	
	#time.sleep(1)
		
