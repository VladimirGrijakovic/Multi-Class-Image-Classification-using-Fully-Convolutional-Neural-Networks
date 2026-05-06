import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras import layers

from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve

from keras.applications import MobileNetV2

training_path = './data/training'
val_path = './data/validation'
test_path = './data/test'
img_size = (160, 160)
batch_size = 64

data_train = image_dataset_from_directory(training_path,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=123)

data_val = image_dataset_from_directory(val_path,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      seed=123)

data_test = image_dataset_from_directory(test_path,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      seed=123)

klase = data_train.class_names
print(f"Klase u treningu: {klase}")
print(f"Broj klasa: {len(klase)}")


def get_labels(dataset):
    return np.concatenate([y for x, y in dataset], axis=0)

y_train = get_labels(data_train)
y_val = get_labels(data_val)
y_test = get_labels(data_test)

train_counts = np.bincount(y_train.astype(int))
val_counts = np.bincount(y_val.astype(int))
test_counts = np.bincount(y_test.astype(int))

labels = ['Trening', 'Validacija', 'Test']
ptice = [train_counts[0], val_counts[0], test_counts[0]]
dronovi = [train_counts[1], val_counts[1], test_counts[1]]

x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, ptice, width, label='Ptice', color='skyblue')
plt.bar(x + width/2, dronovi, width, label='Dronovi', color='salmon')

plt.xticks(x, labels)
plt.ylabel('Broj slika')
plt.title('Koliko slika imamo?')
plt.legend()
plt.show()


img, lab = next(iter(data_train))
plt.figure()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(img[i].numpy().astype('uint8'))
    plt.title(klase[lab[i]])
    plt.axis('off')
plt.show()

data_augmentation = Sequential(
  [
    layers.Input((img_size[0], img_size[1], 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2),
  ]
)

for i in range(10):
    aug_img = data_augmentation(img)
    plt.subplot(2, 5, i+1)
    plt.imshow(aug_img[0].numpy().astype('uint8'))
    plt.axis('off')
plt.show()


def cnn_model(num_classes):
    model = Sequential([
        data_augmentation,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(0.2),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile('adam',
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def transfer_learning(img_size, num_classes):
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = Sequential([
        data_augmentation,
        layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
        base_model,

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

#model = cnn_model(len(klase))
model = transfer_learning(img_size, 2)
model.summary()

es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

y_train = np.concatenate([y for x, y in data_train], axis=0)
counts = np.bincount(y_train.astype(int))
ukupno = len(y_train)

class_weights = {
    0: ukupno / (2 * counts[0]),
    1: ukupno / (2 * counts[1])
}

history = model.fit(data_train,
                    validation_data=data_val,
                    epochs=30,
                    class_weight=class_weights,
                    callbacks=[es],
                    verbose=1
)


model.layers[2].trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
history = model.fit(data_train,
                    validation_data=data_val,
                    epochs=15,
                    class_weight=class_weights,
                    callbacks=[es],
                    verbose=1
)

plt.figure()
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.show()


#matrica konfuzije na test skupu
y_true = np.array([])
y_pred = np.array([])
for img, lab in data_test:
    y_true = np.concatenate([y_true, lab.numpy()])
    preds = model.predict(img, verbose=0)
    y_pred = np.concatenate([y_pred, np.argmax(preds, axis=1)])

print(f"Tačnost modela je: {100 * accuracy_score(y_true, y_pred):.2f}%")

cm = confusion_matrix(y_true, y_pred, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=klase).plot()
plt.show()

#vrednosti performansi klasifikacije na test podacima
cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()
print(f"True Negative: {TN}")
print(f"False Positive: {FP}")
print(f"False Negative: {FN}")
print(f"True Positive: {TP}")

# Tacnost
ACC = (TP + TN) / (TP + FP + TN + FN)
print(f"Tacnost: {ACC:.4f}")

# Preciznost
P = TP / (TP + FP) if (TP + FP) != 0 else 0
print(f"Preciznost: {P:.4f}")

# Osetljivost (Recall)
R = TP / (TP + FN) if (TP + FN) != 0 else 0
print(f"Osetljivost (Recall): {R:.4f}")

# Specificnost
S = TN / (TN + FP) if (TN + FP) != 0 else 0
print(f"Specificnost: {S:.4f}")

# F1-skor
F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0
print(f"F1-skor: {F1:.4f}")

# Balansirana tacnost
BA = (R + S) / 2
print(f"Balansirana tacnost: {BA:.4f}")



#matrica konfuzije na trening skupu
y_true = np.array([])
y_pred = np.array([])
for img, lab in data_train:
    y_true = np.concatenate([y_true, lab.numpy()])
    preds = model.predict(img, verbose=0)
    y_pred = np.concatenate([y_pred, np.argmax(preds, axis=1)])

print(f"Tačnost modela je: {100 * accuracy_score(y_true, y_pred):.2f}%")

cm = confusion_matrix(y_true, y_pred, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=klase).plot()
plt.show()

