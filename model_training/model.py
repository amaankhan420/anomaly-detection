import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications import vgg16
from keras.optimizers import SGD
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns

image_directory = 'C:/Users/user/OneDrive/Desktop/project/planks/'
SIZE = 224
dataset = []
label = []

good_data = os.listdir(image_directory + 'good_data/')
for i, image_name in enumerate(good_data):
    if image_name.split('.')[1] in ['png', 'jpg']:
        image = cv2.imread(image_directory + 'good_data/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

print("gd done")

anomaly_data = os.listdir(image_directory + 'anomaly_data/')
for i, image_name in enumerate(anomaly_data):
    if image_name.split('.')[1] in ['png', 'jpg']:
        image = cv2.imread(image_directory + 'anomaly_data/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

print("ad done")

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.25, random_state=0)

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def get_model(input_shape=(224, 224, 3)):
    vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in vgg.layers[:-5]:
        layer.trainable = False

    x = vgg.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(vgg.input, x)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.0001, momentum=0.9),
                  metrics=['accuracy'])
    return model


model = get_model(input_shape=(SIZE, SIZE, 3))
print("model done")

history = model.fit(x_train, y_train, batch_size=16, epochs=30, verbose=1, validation_data=(x_test, y_test))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

_, acc = model.evaluate(x_test, y_test)
print('Accuracy = ', (acc * 100.0), '%')

n = 1
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
print("The prediction for this image is: ", np.argmax(model.predict(input_img)))
print("The actual label is: ", np.argmax(y_test[n]))

y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
sns.heatmap(cm, annot=True)

model.save('C:/Users/user/OneDrive/Desktop/project/model_training/my_model.h5')
