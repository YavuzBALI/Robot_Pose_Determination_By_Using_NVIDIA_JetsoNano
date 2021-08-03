import cv2
print(cv2.__version__)
import numpy as np
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.python.compiler.tensorrt import trt
import tensorflow as tf
import os

#The trained model is loaded
output_saved_model="/home/yavuz/OpenCv2Tensorflow_RobocupAngle/101x101_image_size/Robot_Angle_pre_SavedModel_101_colab"
saved_model_loaded=tf.saved_model.load(output_saved_model)

#The signature of the model is saved in the variable.
infer =saved_model_loaded.signatures['serving_default']
print(infer.structured_outputs)

#Classes are defined
class_Angle=['Angle0', 'Angle10', 'Angle100', 'Angle105', 'Angle110', 'Angle115', 'Angle120',
'Angle125', 'Angle130', 'Angle135', 'Angle140', 'Angle145', 'Angle15', 'Angle150', 'Angle155',
'Angle160', 'Angle165', 'Angle170', 'Angle175', 'Angle180', 'Angle185', 'Angle190', 'Angle195',
'Angle20', 'Angle200', 'Angle205', 'Angle210', 'Angle215', 'Angle220', 'Angle225', 'Angle230',
'Angle235', 'Angle240', 'Angle245', 'Angle25', 'Angle250', 'Angle255', 'Angle260', 'Angle265', 
'Angle270', 'Angle275', 'Angle280', 'Angle285', 'Angle290', 'Angle295', 'Angle30', 'Angle300', 
'Angle305', 'Angle310','Angle315', 'Angle320', 'Angle325', 'Angle330', 'Angle335', 'Angle340', 
'Angle345', 'Angle35', 'Angle350', 'Angle355', 'Angle40', 'Angle45', 'Angle5', 'Angle50', 'Angle55', 
'Angle60', 'Angle65', 'Angle70', 'Angle75', 'Angle80', 'Angle85', 'Angle90', 'Angle95']

#The initial parameters for FPS are saved
prev_frame_time=0
new_frame_time=0
#Video or camera image is taken.
cam= cv2.VideoCapture('goruntuler/deneme.avi')

'''
Here we specify the HSV code parameters for red and green, respectively.
l_b=[HougeHight_value,Saturation_HightValue,Valuehigh]
u_b=[Hougelow_value,Saturation_LowValue,ValueLow]
'''
l_b=np.array([53,111,167])#For green
u_b=np.array([107,219,241])#For green
l_b2=np.array([129,111,167])#For red
u_b2=np.array([179,255,255])#For red

'''Here it goes into an infinite loop. A frame is taken and processed 
    in each loop until the video ends or the algorithms are stopped'''
while True:
    ret, frame = cam.read()# Assigned each frame to the frame variable

    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#Coverted from BGR format to HSV format for masking.
    FGmask=cv2.inRange(hsv,l_b,u_b)#Applied a filter for the green color
    FGmask2=cv2.inRange(hsv,l_b2,u_b2)# Applied a filter for the red color
    FGmaskComp=cv2.add(FGmask,FGmask2)#Combined filters.
    
    #Here combined the frame and filters in BGR format
    FG=cv2.bitwise_and(frame, frame ,mask=FGmaskComp)

    #Cotours are drawn on the filtred images.
    contours,hierarchy=cv2.findContours(FGmaskComp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)

    for cnt in contours:
        #The area of the drawn contours is drawn.
        area=cv2.contourArea(cnt)
        (x,y,w,h)=cv2.boundingRect(cnt)
        if area>=50:
            #Cropped the robot's position over the masked frame.
            Robot=FG[(y-30):(y+h+30),(x-30):(x+w+30)]
            #Scaled to 101x101 for the model
            Robot_pre=cv2.resize(Robot,(101,101))

            #The colors of the robots are learned.
            color=frame[y,x]
            if color[0]>107:
                Robot_color='Green'
            else:
                Robot_color='Red'
            
            #Pre-processing the images for the model.
            X = image.img_to_array(Robot_pre)
            X = np.expand_dims(X,axis=0)
            images = np.vstack([X])
            image_input=tf.constant(images.astype('float32'))
            #Given the images to the model for inference
            preds = infer(image_input)
            #Taken the maximum of inference.
            result = np.argmax(preds['activation_3'])
            #Written the result of the inference on the image as text on the robots.
            cv2.putText(frame,Robot_color+' '+class_Angle[result],(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    
    # The running time of the algorithm and the FPS paramater are obtained.
    new_frame_time=time.time()
    second=new_frame_time-prev_frame_time
    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time   
    fps=int(fps)
    fps_1=str(fps)
    second_1=str(round(second,4))

    #FPS and processing time were processed as text on the image.
    cv2.putText(frame,fps_1,(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,second_1,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imshow('results',frame)
    
    #It is used to exit the algorithm.
    if cv2.waitKey(1)==ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()
