import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Veri setinin bulunduğu ana klasör yolu
data_dir = 'C:\\Users\\mete1\\OneDrive\\Masaüstü\\Cars Dataset\\train'

# Veri setindeki sınıfları alın
class_names = os.listdir(data_dir)

# Görüntüleri ve etiketleri yüklemek için fonksiyon
def load_images(data_dir, class_names, img_size=(224, 224)):
    images = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                img_path = os.path.join(class_dir, img_file)
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_names.index(class_name))
    return np.array(images), np.array(labels)

# Görüntüleri ve etiketleri yükle
images, labels = load_images(data_dir, class_names)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# Görüntüleri normalleştir
images = images / 255.0

# Etiketleri kategorik hale getir
num_classes = len(class_names)
labels = to_categorical(labels, num_classes)

# Veri setini eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Veri artırma için ImageDataGenerator kullan
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Eğitim verisini artırmak için fit
train_datagen.fit(X_train)

# VGG16 modelini yükle ve transfer öğrenme için kullan
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout ekleyerek overfitting'i azaltabiliriz
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# VGG16 katmanlarının bazılarını ince ayar için aç
for layer in base_model.layers[:15]:
    layer.trainable = False
for layer in base_model.layers[15:]:
    layer.trainable = True

# Modeli derle
optimizer = Adam(learning_rate=0.0001)  # Düşük öğrenme oranı kullan
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks ekle
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğit
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,  
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stopping]
)


model.save('updated_car_classification_model.h5')

# Modeli yükleyin ve derleyin
loaded_model = load_model('updated_car_classification_model.h5')
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli değerlendirin
loss, accuracy = loaded_model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
