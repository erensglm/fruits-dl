
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from glob import glob
import matplotlib.pyplot  as plt

train_path= "C:\\Users\\erens\\OneDrive\\Desktop\\EREN\\DERS\\DerinOgrenmeyeGiris\\fruit-360dataset\\fruits-360_dataset\\fruits-360\Training"
test_path= "C:\\Users\\erens\\OneDrive\\Desktop\\EREN\\DERS\\DerinOgrenmeyeGiris\\fruit-360dataset\\fruits-360_dataset\\fruits-360\\Test\\"
img=load_img(train_path+"Apple Braeburn\\0_100.jpg")
x=img_to_array(img)
print(x.shape)
numberOfClass=len(glob(train_path+'/*'))
print(numberOfClass)

vgg =VGG16()
#print(model.summary())

vgg_layer_list=vgg.layers
print(vgg_layer_list)
print(len(vgg_layer_list))

model =Sequential()

for i in range(len(vgg_layer_list)-1):
    model.add(vgg_layer_list[i])

print(model.summary())

for layers in model.layers:
    layers.trainable=False

model.add(Dense(numberOfClass,activation="softmax"))
print(model.summary())

model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

train_data= ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224))
test_data=ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224))
batch_size=32
hist=model.fit_generator(generator=train_data,steps_per_epoch=20,epochs=2,validation_data=test_data,validation_steps=20)
print(hist.history.keys())
model.save_weights("transfer.h5")



print(hist.history.keys())
plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.title("Loss")
plt.legend()
plt.show()

plt.plot(hist.history["accuracy"],label="Train Accuracy")
plt.plot(hist.history["val_accuracy"],label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()
plt.show()

import json
with open("transfer.json","w") as f:
    json.dump(hist.history,f)






