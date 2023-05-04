import streamlit as st #streamlit==1.22.0
import matplotlib.pyplot as plt #matplotlib==3.7.1
import requests #requests==2.29.0
import numpy as np #numpy==1.23.5  
from torchvision.io import read_image   #torchvision==0.15.1

import os
from keras.models import Sequential  #keras==2.12.0
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import cv2  #opencv-python==4.7.0.72
from PIL import Image
from pathlib import Path
import tempfile
import datetime

# 画像ファイルのパス
from tensorflow.python.keras.layers import Activation #tensorflow==2.12.0

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像判定アプリ")
st.sidebar.write("画像より判定します。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像ファイルを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
    
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

###
###
###
if img_file is not None:
    # 画像を表示する
    st.image(img_file)

    # ファイルパス取得
    t_dalta = datetime.timedelta(hours=9)
    JST =datetime.timezone(t_dalta,'JST')
    now = datetime.datetime.now(JST)
    d = now.strftime('%Y%m%d%H%M%S')
    file_dir = "./data"
    file_temp = os.path.join(file_dir, "temp.png")
    file_name =  str(d) + "_" + img_file.name
    file_path = os.path.join(file_dir, file_name)

    #print('@@@@@@@@@@@-Start-@@@@@@@@@@')
    #print(img_file)
    #print(d)
    #print(file_name)
    print(file_path)
    #print('@@@@@@@@@@@- End -@@@@@@@@@@')

    # ファイルOpen,Save
    img = Image.open(img_file)
    img.save(file_path)

    IMG_SIZE = 64

    # モデルの作成とトレーニング
    data = []
    labels = []
    classes = ['lighton', 'lightoff']

    for c in classes:
        path = os.path.join(os.getcwd(), c)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(classes.index(c))
                print(img_path)
            except Exception as e:
                print(e)

    data = np.array(data)
    labels = np.array(labels)

    # データをシャッフルする
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]

    # データをトレーニング用と検証用に分割する
    num_samples = len(data)
    num_train = int(num_samples * 0.8)
    x_train = data[:num_train]
    y_train = labels[:num_train]
    x_val = data[num_train:]
    y_val = labels[num_train:]

    # 画像データの正規化
    x_train = x_train / 255.0
    x_val = x_val / 255.0

    # ラベルをone-hotエンコーディングに変換する
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # モデルを構築する
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # モデルをコンパイルする
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    # モデルをトレーニングする
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

    # 画像を選択する
    #file_path = img_path
    #print(img_path)
    #print(file_path)

    # 選択された画像をモデルに入力して予測結果を表示する
    image_selected = cv2.imread(file_path)
    image_selected = cv2.resize(image_selected, (IMG_SIZE, IMG_SIZE))
    image_selected = np.expand_dims(image_selected, axis=0)
    image_selected = image_selected / 255.0
    prediction = model.predict(image_selected)
    if np.argmax(prediction) == 0:
        st.write("LIGHT-ON!")
        print("LIGHT-ON")
    else:
        st.write("LIGHT-OFF!")
        print("LIGHT-OFF")
