import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
base_path = r'C:\Users\hp\Desktop\Jainil\photos'
temp_base = r'C:\Users\hp\Desktop\Jainil\photos_sampled'

train_dir = os.path.join(temp_base, 'train')
val_dir = os.path.join(temp_base, 'val')

if os.path.exists(temp_base):
    shutil.rmtree(temp_base)

os.makedirs(train_dir)
os.makedirs(val_dir)
for class_name in os.listdir(base_path):
    class_path = os.path.join(base_path, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)
    selected = images[:25]
    train_imgs = selected[:20]
    val_imgs = selected[20:]

    os.makedirs(os.path.join(train_dir, class_name))
    os.makedirs(os.path.join(val_dir, class_name))

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
img_size = (128, 128)
batch_size = 8

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=5)
model.save('celebrity_cnn_model.h5')