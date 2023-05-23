import os
import time
import serial
import cv2 as cv
import file_read
import dijkstra

#down = "python3 downimg.py"
#os.system(down)
#time.sleep(1)

#Serial Setup
py_serial = serial.Serial(port = '/dev/ttyUSB0', baudrate = 9600)


#Camera Setup
#webCam.set(cv.CAP_PROP_FRAME_WIDTH, 1000) #width = 2560px
#webCam.set(cv.CAP_PROP_FRAME_HEIGHT, 720) #height = 1440px

start_row = 0
start_col = 0
check = 0

while True :
	picture = "fswebcam --no-banner --set brightness=60% Images/test1.jpg"
	os.system(picture)
	img = cv.imread("Images/test1.jpg", cv.IMREAD_COLOR)
	resize_img = cv.resize(img, (1020,720), interpolation=cv.INTER_AREA)
	cv.imwrite("Images/test1.jpg", resize_img)
	#Image Analysis
	yolo = "python3 /home/sgme/yolov5/detect.py > /home/sgme/yolov5/output.txt --weights /home/sgme/yolov5/best.pt --img 640 --conf 0.4 --source /home/sgme/Images/test1.jpg"
	os.system(yolo)
	time.sleep(1)
	#Check object
	lines = open('/home/sgme/yolov5/output.txt').readlines()

	if check == 2:
	    break

	if len(lines)<=1:
		check += 1
		continue
		    
	else:
		#Make Map using object coordination
		check = 0
		given_map=file_read.map()
		    
		#Make move order from start position
		move_order, end_row, end_col = dijkstra.main(given_map, start_row, start_col)
		#print map
		for i in range(len(given_map)) :
			print(given_map[i])
		print(move_order)
		    
		#msg = "go"
		#py_serial.write(msg.encode())
		#time.sleep(2)
	    
		while len(move_order):
			py_serial.write(move_order.pop(0))
			print("pop")
			time.sleep(2) #delay time decided by velocity and distance
			#return_msg = py_serial.readline().decode('utf-8')
			#print(return_msg)
			#while(return_msg[0:4] != "done"):
			    #return_msg = py_serial.readline().decode('utf-8')
			    #print(return_msg)
			    #time.sleep(0.5)
		time.sleep(5)
		start_row = end_row
		start_col = end_col
	  
