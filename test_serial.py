import serial
import time

py_serial = serial.Serial(port = '/dev/ttyUSB2', baudrate = 9600, timeout = 5 )

distance = [(1, 10, 1, 10), (0,10,0,10), (1, 10, 1, 10)]
#distance = [(0, 10, 0, 10), (0, 10, 0, 10), (0, 10, 0, 10), (1, 10, 1, 10), (0, 10, 0, 10), (1, 10, 1, 10), (0, 10, 0, 10), (1, 10, 1, 10), (0, 10, 0, 10), (1, 10, 1, 10), (0, 10, 0, 10), (1, 10, 1, 10), (0, 10, 0, 10), (1, 10, 1, 10), (0, 10, 0, 10), (1, 10, 1, 10), (0, 10, 0, 10)]

while len(distance):
    py_serial.write(distance.pop(0))
    print("pop")
    time.sleep(0.1) #delay time decided by velocity and distance
    
    
