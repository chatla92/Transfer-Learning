from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import pdb

import cv2

import matplotlib.pyplot as plt
import random


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

random.seed(8675309)

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

from keras.callbacks import ModelCheckpoint  

num_epochs = 50
batch_size = 64
bottlenck_file = 'bottleneck_features/DogInceptionV3Data.npz'
bottleneck_features = np.load(bottlenck_file)

# Precomputed feature values
train_network = bottleneck_features['train']
valid_network = bottleneck_features['valid']
test_network = bottleneck_features['test']

# Model Design
network_model = Sequential()
network_model.add(GlobalAveragePooling2D(input_shape=train_network.shape[1:]))
#network_model.add(Dropout(0.3))
#network_model.add(Dense(256, activation='relu'))
network_model.add(Dropout(0.2))
network_model.add(Dense(133, activation='softmax'))
network_model.summary()

# Compile the Model
network_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Checkpoint to save the model with least loss
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.network.hdf5',
				verbose=1, save_best_only=True)
#Traning the Model
network_model.fit(train_network, train_targets,
		validation_data=(valid_network, valid_targets),
	         epochs=num_epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
network_model.load_weights('saved_models/weights.best.network.hdf5')

# Predictions and accuracy caclculation
network_predictions = [np.argmax(network_model.predict(np.expand_dims(feature, axis=0))) for feature in test_network]
test_accuracy = 100*np.sum(np.array(network_predictions)==np.argmax(test_targets, axis=1))/len(network_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

dog_files_short = train_files[:10]
from extract_bottleneck_features import *
from keras.preprocessing import image  

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

for img_path in dog_files_short:
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    
    if 'Inception' in bottlenck_file:
        bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    elif 'VGG16' in bottlenck_file:
        bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    elif 'VGG19' in bottleneck_file:
        bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    elif 'Xception' in bottleneck_file:
        bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    elif 'Resnet' in bottleneck_file:
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = network_model.predict(bottleneck_feature)
    breed = ("This dog belongs to " + str(dog_names[np.argmax(predicted_vector)]) + " breed")
    plt.title(breed)
    plt.show()
