
# Robot_Pose_Determination_By_Using_NVIDIA_JetsoNano
This repo contains the codes and explanations of my graduation thesis that I carried out at Dokuz Eylül University in 2021.

# Abstract

   The RoboCup football league is a robotics and artificial intelligence workspace that has received a lot of attention. In the RoboCup Small Size League (SSL), one of the RoboCup 
leagues, football is played with 2 teams consisting of more than one robot. Robots are expected to act autonomously. In this field of study, strategy software, artificial 
intelligence, computer vision, etc. studies are included.

   Real-time and accurate robot pose identification is one of the most important issues for RoboCup. In this project, the robot pose to be used in RoboCup SSL competitions has 
been examined. The pose contains the position and angle data of a robot. Image processing methods have been evaluated to find the position of the robot. Deep learning algorithms 
are used to access angle information of robots with positions. Deep learning algorithms have quite a large computational load. For this reason, Jetson Nano Developer kit was used 
in the project in order to be able to calculate deep learning algorithms and provide high performance inference. A real-time game is in progress in the RoboCup football league. 
Therefore, robot pose detection with high precision in the lowest processing time is aimed in the project.

   ![1](https://user-images.githubusercontent.com/84620286/128405283-802e0e4c-8617-4bd6-aa5f-97330be3fbfd.PNG)

# Summary
- Robot detection was found by color masking with the help of OpenCV library.
- Robot angles were determined with the help of Deep learning algorithms.
- The color and angle of the robots are displayed at the top right of the robots.
- The dataset consists of 1440 manually generated images for 72 classes.
- Jetson Nano 4 GB was used as the processor.
- Colab used for training

# Table of Content
1. Robot Position Detection
2. Model and Dataset
3. Training and Evaluate
4. Perform
5. My Observations About the Task
6. Discussion

# 1. Robot Position Detection

   The filtering method was used to determine the robot position. Thanks to the colors of the circles on the robots, it allows us to determine their positions. Robot detection
is made for 2 different colors. It includes the colors green and red. The colors of the robots also form the identity of the robots. First, the image taken from the camera is
converted from RGB format to HSV format. Then, 2 different maskings in green and red are applied. HSV values of colors are required for masking. HSV values are different for
each color. Table  shows the HSV thresholds for colors.Then, the values of the robot positions in the coordinate plane can be obtained by applying the contour to the filtered
images.
![HSV_Treshold](https://user-images.githubusercontent.com/84620286/128512685-c3c2e88c-f034-4941-a21b-456a4f0f465a.PNG)

# 2. Model and Dataset
Deep learning algorithms were used to determine the angles of the robots. Robots whose positions are known are cropped to size, including only one robot, and
given as input to the model for prediction. The model was created using the CNN architecture. The RoboCup-1 model used in Ayşe Ezgi Kabukuçu's project "RoboCup
Robot Localization by Using Deep Learning Principle" is based. This model was developed to achieve high FPS. The RoboCup-III model was presented as a solution.

![RoboCup-III](https://user-images.githubusercontent.com/84620286/128514513-46740e47-7e55-4965-b651-3ade52ba34ae.PNG)

The images created by rotating the images taken manually at 0 degrees with a difference of 5 degrees were used as the data set. A total of 20 images were created at 0 degrees. A
total of 1440 images were created for 72 classes. 80% of these images were used for training and 20% for validation.

![Example of dataset](https://user-images.githubusercontent.com/84620286/128515873-ddf5d9dc-f856-4f6a-a06e-c0b2231d1726.PNG)

# 3. Training and Evaluate

The new model RoboCup-III was trained with a dataset of 1440 images. Training was carried out using 8 batch sizes and 18 epochs. The learning rate was chosen as 0.0001. Adam
optimizer was used. The following images show the loss, accuracy graph and the final evolution metric value table of the model, respectively.

![Loss Graph](https://user-images.githubusercontent.com/84620286/128518817-ac8c2ad1-ed11-4a05-bd92-4ef57a204f02.PNG)
![Accuracy Graph](https://user-images.githubusercontent.com/84620286/128518640-867e4df8-5f3d-4424-8f36-470399be07c9.PNG)

![Evaluate Table](https://user-images.githubusercontent.com/84620286/128518659-c6796bb5-be0b-49f2-9fb5-aeb68403f358.PNG)


# 4. Perform
Prediction was done on Jetson Nano Developer kit. Weight parameters were set to FP16 and FP32 parameters using the TensorRT library.You can find the conversion code at Convert_TF-TRT.py . The FPS values of the model in different
parameters are given in the table below.

![Perform Table](https://user-images.githubusercontent.com/84620286/128520129-f30fe547-c037-4d1c-886a-0c38cc59f81f.PNG)

# 5. My Observations About the Task

When the performance values are examined, it is seen that the effect of the TensorRT library in small models is small. In addition, the use of vehicle location detection as a
filtering method and angle detection as a deep learning algorithm causes a decrease in FPS. YOLO etc. By solving both problems in a single source by using algorithms, results
can be obtained in higher FPS parameters. In addition, with more data, the degree differences can be reduced. Angle estimation can be made with 1, 2 or 3 degree differences.

# 6. Discussion

If you have any question and suggestion, please do not hesitate to contact me.

# Thanks!!!!
