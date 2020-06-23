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

        
  Next , is to input images . since , we have limited images therefore before giving to model we need feature engineering to bo done. So , we imported Imagedatagenerator to do feature engineering.
  
       from keras_preprocessing.image import ImageDataGenerator
       train_data_dir = 'F:/MLOPS/practise/ENVIRONMENT2/ankush_dataset_images/train/'
       validation_data_dir = 'F:/MLOPS/practise/ENVIRONMENT2/ankush_dataset_images/test/'


       train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=0.3,
                    height_shift_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest')
 
       validation_datagen = ImageDataGenerator(rescale=1./255)
 
       batch_size = 32
 
       train_generator = train_datagen.flow_from_directory(
                     train_data_dir,
                     target_size=(224,224),
                     batch_size=batch_size,
                     class_mode='categorical')
 
       validation_generator = validation_datagen.flow_from_directory(
                     validation_data_dir,
                     target_size=(224,224),
                     batch_size=batch_size,
                     class_mode='categorical')
                     
                     
 The final step is to compile the model and then start training the model . So , we used adam optimizer and the binary_crossentropy for calculating the loss . Next we trained the model with defined no of epochs and provide the images.
 
        from keras.optimizers import RMSprop
        from keras.optimizers import Adam
        from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
       checkpoint = ModelCheckpoint("face_detection.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

        earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
       callbacks = [earlystop, checkpoint]

# We use a very small learning rate 
        new_model.compile(loss = 'binary_crossentropy',
                     optimizer = Adam(lr = 0.0001),
                      metrics = ['accuracy'])

# Enter the number of training and validation samples here
        nb_train_samples = 2398
        nb_validation_samples = 72

# We only train 3 EPOCHS 
        epochs = 3
        batch_size = 32

        history = new_model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size)
        
        
![Screenshot (492)](https://user-images.githubusercontent.com/51692515/85364381-a476b980-b540-11ea-8b45-eeb69f8c7ee6.png)


# NEXT IS TO TEST OUR MODEL :
  
# so we created code for video streaming and identify the person in it:
  The code for same is also present in the repo above.
  
  Firstly , we loaded the required modules for video streaming , loading model and image for preprocessing purpose.
     
           import cv2
           import numpy
           from keras.models import load_model
           from keras.preprocessing import image
   
  Next , we loaded the trained model and activated the camera to start video stream.
  
           check_model=load_model('face_detection.h5')
           camera=cv2.VideoCapture(0)
 
 Next , we created while loop for continous stream , read the photo and changed the size of image as defined by our model . NOTE that VGG19 requires 4D images thats why we changed the Dimensions of the image. we predicted the output and applied the if conditions to further filter it.
               
            while 1:
                status,photo=camera.read()
                test_image=cv2.resize(photo,dsize=(224,224), interpolation=cv2.INTER_CUBIC)
                image_test=numpy.expand_dims(test_image,axis=0)
    
                output=check_model.predict(image_test)
                #print(check_model.predict(image_test))

                if output[0][0]>= output[0][1] :
                     print(check_model.predict(image_test))
                     texted_image =cv2.putText(img=photo, text="ANKUSH", org=(50,400),fontFace=2, fontScale=2, color=(0,0,255), thickness=3)
               elif output[0][0] < output[0][1] :
                      print(check_model.predict(image_test))
                      texted_image =cv2.putText(img=photo, text="ARTI", org=(50,400),fontFace=2, fontScale=2, color=(0,255,0), thickness=3)
        
               cv2.imshow("VIDEO",texted_image)
               cv2.waitKey(10)
               if cv2.waitKey(100) == 13:
                      break
        
        
       cv2.destroyAllWindows()
       camera.release()
       
       
# OUTPUT:

# My model was successfully able to identify me.

![Screenshot (493)](https://user-images.githubusercontent.com/51692515/85365288-ae012100-b542-11ea-8a22-ae91d23553dc.png)


