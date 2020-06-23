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
  We used VGG19 model to train our model (VGG19 is a predefined powerfull model that has 19 layers out of which 16 are convolve relu pooling and 3 are FC).The input shape is given to define the size of photo that is to be input and include top is defined to remove the last softmax layer .
  
        from keras.applications import VGG19
        model=VGG19(weights='imagenet',include_top=False,input_shape=(224,224,3))
        
 Next, is to change all the layers that can be edited to false.
        
        for layers in model.layers:
        layers.trainable=False

Next is to import all the required modules to create new dense layers where we will give neurons so, as to train the model.
so we imported Dense layer , sequential model , adam optimiser and pooling2d layer.

        from keras.layers import Dense
        from keras.models import Sequential
        from keras.optimizers import Adam
        from keras.layers import GlobalAveragePooling2D
 
Next , step is to actualy create the dense layers . so we created the layers with the respective neurons as mentioned and always attached to the last layer.

      last_layer=GlobalAveragePooling2D()(last_layer)
      last_layer=Dense(512,activation='relu')(last_layer)
      last_layer=Dense(250,activation='relu')(last_layer)
      last_layer=Dense(100,activation='relu')(last_layer)
      last_layer=Dense(50,activation='relu')(last_layer)
      last_layer=Dense(10,activation='relu')(last_layer)
      last_layer=Dense(2,activation='sigmoid')(last_layer)
      from keras.models import Model
      new_model=Model(inputs=model.input , outputs=last_layer)
      
 ![Screenshot (490)](https://user-images.githubusercontent.com/51692515/85363558-d1c26800-b53e-11ea-85db-fba85a3c0d75.png)
 
![Screenshot (491)](https://user-images.githubusercontent.com/51692515/85363562-d424c200-b53e-11ea-9cc5-8b42a739322d.png)

        
  
