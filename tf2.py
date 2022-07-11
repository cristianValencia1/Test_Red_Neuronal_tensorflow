from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, Sequential
from tensorflow.keras.layers import Flatten, Dense

modelo = Sequential()

modelo.add(Flatten(input_shape=(28, 28)))
modelo.add(Dense(128, activation='relu'))
modelo.add(Dense(10, activation='softmax'))

modelo.compile(optimizer='adam',
               loss='categorical_crossentropy', metrics=['accurancy'])

modelo.fit(x_train, y_train, epochs=10, verbose=1)

prediciones = modelo.predict(x_test)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

modelo = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')

])

modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accurancy'])


modelo.fit(x_train, y_train, epochs=5)

modelo.evaluate(x_test, y_test, verbose=2)
== == == == == == == == == == == == == == == == == == == == == == == =


inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()

== == == == == == == == == == == == == == == == == == == == == == == ==

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model.fit(xs, ys, epochs=50)

print(model.predict([10.0]))

== == == == == == == == == == == == == == == == == == ==


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=[]):
        if(logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images/255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

test_loss, test_accuracy = model.evaluate(test_images, test_labels)


print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))
