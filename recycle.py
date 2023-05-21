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
py_serial = serial.Serial(port = '/dev/ttyUSB1', baudrate = 9600)

#Open webCam
webCam = cv.VideoCapture(0)
if not webCam.isOpened():
    print("Can't open camera")	
    exit()
    
#Camera Setup
#webCam.set(cv.CAP_PROP_FRAME_WIDTH, 1000) #width = 2560px
#webCam.set(cv.CAP_PROP_FRAME_HEIGHT, 720) #height = 1440px

start_row = 0
start_col = 0

#Take picture
usable, img = webCam.read()
if not usable:
    print("Can't recieve img")
else:
    resize_img = cv.resize(img, (1020,720), interpolation=cv.INTER_CUBIC)
    cv.imshow("webCam", resize_img)
    while True:
	    #Save Image
	    cv.imwrite("/home/sgme/Images/test1.jpg", resize_img)
	    time.sleep(1)
	    #Image Analysis
	    yolo = "python3 /home/sgme/yolov5/detect.py > /home/sgme/yolov5/output.txt --weights /home/sgme/yolov5/best.pt --img 640 --conf 0.3 --source /home/sgme/Images/test1.jpg"
	    os.system(yolo)
	    time.sleep(1)
	    #Check object
	    lines = open('/home/sgme/yolov5/output.txt').readlines()
	    #Case : Zero object
	    if len(lines)<=1:
		    print("There is no remain object!")
		    break
	    #Case : Object exist
	    else:	
		    #Make Map using object coordination
		    given_map=file_read.map()
		    #print map
		    for i in range(len(given_map)) :
			    print(given_map[i])
		    #Make move order from start position
	    
		    move_order, end_row, end_col = dijkstra.main(given_map, start_row, start_col)
		    
		    print(move_order)
		    
		    
		    
		    #msg = "go"
		    #py_serial.write(msg.encode())
		    time.sleep(2)
	    
		    while len(move_order):
			    py_serial.write(move_order.pop(0))
			    print("pop")
			    time.sleep(0.5) #delay time decided by velocity and distance
		
		    return_msg = py_serial.readline().decode('utf-8')
		    print(return_msg)
		    
		    while(return_msg[0:4] != "done"):
			    print(return_msg)
			    time.sleep(1)
		    time.sleep(4)
		    start_row = end_row
		    start_col = end_col
	    
webCam.release()
cv.destroyAllWindows()
