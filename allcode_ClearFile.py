""" ラベリングによる学習/検証データの準備 """
import glob
import numpy as np
import random, math
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import np_utils
from keras.models import model_from_json
from keras_preprocessing.image import img_to_array, load_img, array_to_img
from keras.preprocessing import image
from google.colab.patches import cv2_imshow
import pandas as pd
import cv2
from keras import backend as K

# 画像用
from keras.preprocessing.image import array_to_img, img_to_array, load_img
# モデル読み込み用
from keras.models import load_model
# Grad−CAM計算用
from tensorflow.keras import models
import tensorflow as tf

def image_reshape(fname):
    """ 渡されたデータを読み込んで整形して出力"""
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((197, 197))
    data = np.asarray(img)
    return data

def split_dataset(dataset):
    data = [image_reshape(dataset[i][1]) for i in range(len(dataset))]
    label = [dataset[i][0] for i in range(len(dataset))]
    return data, label

def make_sample(categories, root_dir='.', train_ratio=0.8):
    """ 画像とカテゴリから訓練データと検証データを生成 """
     
    if categories is None:
        raise TypeError("categoriesにはリストで値を入れてください。")
  
    """ カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる """
    list_files = []
    for idx, category in enumerate(categories):
        image_dir = root_dir + "/" + category
        files = glob.glob(image_dir + "/*.png")
        for f in files:
            list_files.append((idx, f))
    
    """ シャッフル後、学習データと検証データに分ける """
    random.shuffle(list_files)
    th = math.floor(len(list_files) * train_ratio)
    train = list_files[0:th]
    test  = list_files[th:]
    
    train_data, train_label = split_dataset(train)
    test_data, test_label = split_dataset(test)

    return train_data, train_label, test_data, test_label,train,test

def model_create():
    from keras import layers, models
    from keras import optimizers
    """ モデルの構築 """
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(197,197,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))  #過学習を防ぐ
    model.add(layers.Dense(512,activation="relu"))
    model.add(layers.Dense(2,activation="sigmoid")) #分類先の種類分設定
    
    """ モデル構成の確認 """
    model.summary()
    
    """ モデルのコンパイル """
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=["acc"])
    
    return model

def print_answer(features, correct_label,print_img):
    #予測結果によって処理を分ける
    for i,ans in enumerate(features):
        if (ans.argmax()==1) and (correct_label[i].argmax()==1):
            print ("ファイル名:"+str(print_img[i])+"予測：良品 、データ:良品、OK")
    
        elif (ans.argmax()== 1) and (correct_label[i].argmax()==0):
            print ("ファイル名:"+str(print_img[i])+"予測：良品 、データ:不良品、NG" )
    
        elif (ans.argmax()==0) and (correct_label[i].argmax()==1):
            print ("ファイル名:"+str(print_img[i])+"予測：不良品、データ:良品、NG")
            
        elif (ans.argmax()==0) and (correct_label[i].argmax()==0):
            print ("ファイル名:"+str(print_img[i])+"予測：不良品 、データ:不良品、OK")
        
        else:
            print("error")

def main1():
    #画像が保存されているルートディレクトリのパス
    root_dir = "Model"
    # 商品名
    categories = ["不良品","良品"]
    nb_classes = len(categories)
    
    train_data, train_label, test_data, test_label,train,test = make_sample(categories, root_dir)
    xy = (train_data, train_label, test_data, test_label)

    #データを保存する（データの名前を「tea_data.npy」としている）
    np.save("ClearFile_data.npy", xy)
    
    #データの正規化
    train_data = np.array(train_data).astype("float") / 255
    test_data  = np.array(test_data).astype("float") / 255
    
    #kerasで扱えるようにcategoriesをベクトルに変換
    train_label = np_utils.to_categorical(train_label, nb_classes)
    test_label  = np_utils.to_categorical(test_label, nb_classes)
    
    #モデルの構築
    model = model_create()

    #モデルの学習
    result = model.fit(train_data,
                      train_label,
                      epochs=10,
                      batch_size=8,
                      validation_data=(test_data,test_label))    
    
    acc = result.history['acc']
    val_acc = result.history['val_acc']
    loss = result.history['loss']
    val_loss = result.history['val_loss']
    epochs = range(len(acc))
    
    plt.figure()
    plt.ylim([0.7,1.01])
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('ClearFile_accuracy_graph')
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('ClearFile_loss_graph')
    
    #モデルの保存
    json_string = model.to_json()
    open('ClearFile_modelResult.json', 'w').write(json_string)
    
    #重みの保存
    hdf5_file = "ClearFile_modelResult.hdf5"
    model.save_weights(hdf5_file)

    
    score = model.evaluate(x=train_data,y=train_label,batch_size=8)

    score = model.evaluate(x=test_data,y=test_label,batch_size=8)

    # train.sort()
    # test.sort()
    print("---- 訓練データの損失・精度確認 ----")
    print('loss=', score[0])
    print('accuracy=', score[1])
    print_answer(model.predict(train_data), train_label,train)
    print("---- 検証データの損失・精度確認 ----")
    print_answer(model.predict(test_data), test_label,test)
    
    #確認
    # good_img = []
    # files = glob.glob("Model/良品/*.png")
    # for f in files:
    #     good_img.append((1, f))
    
    # correct_data, correct_label = split_dataset(good_img)
    # correct_label = np_utils.to_categorical(correct_label, nb_classes)
    # correct_data = np.array(correct_data).astype("float") / 255
    
    # #予測
    # score = model.evaluate(x=correct_data,y=correct_label,batch_size=8)
    
    # print("---- 良品データの損失・精度確認 ----")
    # print('loss=', score[0])
    # print('accuracy=', score[1])
    # print_answer(model.predict(correct_data), correct_label,good_img)

    #未知のテストデータ
    # good_img = []
    # files = glob.glob("TEST/不良品/*.png")
    # for f in files:
    #     good_img.append((0, f))
    
    # correct_data, correct_label = split_dataset(good_img)
    # correct_label = np_utils.to_categorical(correct_label, nb_classes)
    # correct_data = np.array(correct_data).astype("float") / 255
    
    # #予測
    # score = model.evaluate(x=correct_data,y=correct_label,batch_size=8)
    
    # print("---- 未知データの損失・精度確認 ----")
    # print('loss=', score[0])
    # print('accuracy=', score[1])
    # print_answer(model.predict(correct_data), correct_label,good_img)

if __name__ == '__main__':
    main1()
