from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# class sayısı için
from glob import glob

import matplotlib.pyplot as plt

train_path = "C:\\Users\\erens\\OneDrive\\Desktop\\EREN\\DERS\\DerinOgrenmeyeGiris\\fruit-360dataset\\fruits-360_dataset\\fruits-360\\Training"
test_path = "C:\\Users\\erens\\OneDrive\\Desktop\\EREN\\DERS\\DerinOgrenmeyeGiris\\fruit-360dataset\\fruits-360_dataset\\fruits-360\\Test"

img= load_img(train_path+ "Apple Braeburn/0_100.jpg")
plt.imshow(img)
plt.axis('off')
plt.show()


