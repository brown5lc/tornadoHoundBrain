from keras.applications import VGG16
from keras import Model 
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Load the VGG model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False 
    
#Add custom layers on top of the VGG model
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

# Create the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators for feeding training/validation images to the model
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('./data_set/training_data', target_size=(224, 224), batch_size=32, class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory('./data_set/validation', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Train the model
history = model.fit(train_generator, steps_per_epoch= 800 // 32,
                    validation_data=validation_generator,
                    validation_steps= 150 // 32, epochs=20)

model.save('./trained_model', save_format='tf')