import serial
import time

py_serial = serial.Serial(port = '/dev/ttyUSB0', baudrate = 9600)

distance = [(0, 30, 6), (0, 30, 2), (5, 0, 4)]
iteration = 0
#msg = "go"
#py_serial.write(msg.encode())
time.sleep(2)
while len(distance):
    py_serial.write(distance.pop(0))
    print("pop")
    
    time.sleep(0.5) #delay time decided by velocity and distance
    iteration += 1
    
