import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import requests
from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation, Flatten, Dense
import mlflow
import mlflow.keras
import argparse

DATASET = "dataset"
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 8
EPOCHS = 10

@task
def download():
    for path in [
        "dataset/train/apples",
        "dataset/val/apples",
        "dataset/test/apples",
        "dataset/train/oranges",
        "dataset/val/oranges",
        "dataset/test/oranges",
    ]:
        os.makedirs(path, exist_ok=True)


    def download_from_list(list, type):
        for i, img_url in enumerate(list):
            response = requests.get(img_url)
            response.raise_for_status()

            ml_split = "train"
            if i == 9:
                ml_split = "test"
            elif i == 8:
                ml_split = "val"

            with open(f"dataset/{ml_split}/{type}s/{type}{i}.jpeg", "wb") as file:
                file.write(response.content)


    download_from_list(
        [
            "https://images.unsplash.com/photo-1570913149827-d2ac84ab3f9a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8MXx8fGVufDB8fHx8fA%3D%3D&w=1000&q=80",
            "https://thumbs.dreamstime.com/b/red-apple-isolated-clipping-path-19130134.jpg",
            "https://images.unsplash.com/photo-1610397962076-02407a169a5b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8Mnx8fGVufDB8fHx8fA%3D%3D&w=1000&q=80",
            "https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2940&q=80",
            "https://images.unsplash.com/photo-1576179635662-9d1983e97e1e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2787&q=80",
            "https://domf5oio6qrcr.cloudfront.net/medialibrary/11525/0a5ae820-7051-4495-bcca-61bf02897472.jpg",
            "https://img.freepik.com/free-photo/two-red-apples-isolated-white_114579-73124.jpg",
            "https://t3.ftcdn.net/jpg/01/09/81/46/360_F_109814626_y5dGATGj8h3pMz9tq1HNRfiuXR12uFCj.jpg",
            "https://i.pinimg.com/originals/e7/4e/78/e74e782a805bf6f2cc8f178a6063f9d7.jpg",
            "https://images.unsplash.com/photo-1584306670957-acf935f5033c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8OHx8fGVufDB8fHx8fA%3D%3D&w=1000&q=80",
        ],
        "apple",
    )

    download_from_list(
        [
            "https://plus.unsplash.com/premium_photo-1671013032586-3e9a5c49ce3c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2787&q=80",
            "https://images.unsplash.com/photo-1547514701-42782101795e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2787&q=80",
            "https://images.unsplash.com/photo-1514936477380-5ea603b9a1ca?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2835&q=80",
            "https://img.freepik.com/free-photo/orange-white-white_144627-16571.jpg?w=2000",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWbb0dC-vAS3Mqx6l_F6uDkUSWFtjHJ8v-MA&usqp=CAU",
            "https://static3.depositphotos.com/1000955/120/i/450/depositphotos_1207359-stock-photo-orange.jpg",
            "https://publish.purewow.net/wp-content/uploads/sites/2/2021/02/types-of-oranges-navel-oranges.jpg?fit=680%2C489",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQnS7lPkp0iYsdVraHtEdmNMQ4g7CFNXGZIuFPyNDZamQG29q6K2mLKo1MbSeYfn8NdWoM&usqp=CAU",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSgA7xpOViT-HeWGzj7f3-rWgX9Fu-dabTj4g&usqp=CAU",
            "https://www.tastingtable.com/img/gallery/the-science-behind-seedless-oranges/l-intro-1655473463.jpg",
        ],
        "orange",
    )
@task
def preprocess():
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        f"{DATASET}/train",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        f"{DATASET}/val",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
    )

    test_generator = test_datagen.flow_from_directory(
        f"{DATASET}/test",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
    )
    return train_generator, validation_generator, test_generator

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))

    return model


@flow
def train():

    model = build_model()
    train_generator = preprocess()[0]
    validation_generator = preprocess()[1]

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    _ = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
    )
    return model

@flow
def eval(model):
    test_generator = preprocess()[2]
    test_loss, test_accuracy = model.evaluate(
        test_generator,
        steps=len(test_generator)
    )

    print('Test accuracy:', test_accuracy)
    print('Test Loss:', test_loss)


@flow(task_runner=SequentialTaskRunner(), log_prints=True)
def main():
    download()
    preprocess()
    model = train()
    eval(model)

if __name__ == '__main__':
    main()