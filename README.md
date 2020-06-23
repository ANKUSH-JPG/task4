# task4
# TRAINING MODEL FOR FACE DETECTION 

# TASK COMPLETED:
Problem Statement: Create a project using transfer learning solving various problems like Face Recognition, Image Classification, using existing Deep Learning models like VGG16, VGG19, ResNet, etc.

# TASK IN ACTION:
 
# creating the dataset:
  The first step is to create the dataset , that has images needed by model to train itself . we divided the images to train dataset and test dataset.
  For creating the images dataset we created a code in python , that will activate the camera and click the desired no of photos for us.
  The code for same is present in repo above .
  
![Screenshot (486)](https://user-images.githubusercontent.com/51692515/85361992-ffa5ad80-b53a-11ea-8d28-524a5196df1f.png)

Here in the code we imported cv2 for clicking photos , activated the camera , and saved the photos at desired place.

NOW LETS HAVE LOOK AT THE DATASET CREATED:

![Screenshot (487)](https://user-images.githubusercontent.com/51692515/85362209-8195d680-b53b-11ea-8386-fa7703be467a.png)

![Screenshot (488)](https://user-images.githubusercontent.com/51692515/85362221-88244e00-b53b-11ea-9e5a-3972739cdf1d.png)

![Screenshot (489)](https://user-images.githubusercontent.com/51692515/85362239-92464c80-b53b-11ea-8275-78aee28096f4.png)

# model training:
Now lets , have a look at the code to train our model using transfer learning .
