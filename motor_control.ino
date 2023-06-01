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
