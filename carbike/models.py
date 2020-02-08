from __future__ import absolute_import, division, print_function, unicode_literals
from django.db import models
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from PIL import Image
import io, base64

graph = tf.get_default_graph()

# Create your models here.
class Photo(models.Model):
    image = models.ImageField(upload_to='photos')

    IMAGE_SIZE =224#画像サイズ
    MODEL_FILE_PATH = './carbike/ml_models/vgg16_transfer.h5'
    classes = ["car", "motorbike"]
    num_classes = len(classes)

    #引数から画像ファイルを参照して読み込む
    def predict(self):
        model = None#モデルの初期化
        global graph
        with graph.as_default():
            model = load_model(self.MODEL_FILE_PATH)
            
            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)

            image = Image.open(img_bin)
            image = image.convert("RGB")
            image = image.resize((self.IMAGE_SIZE,self.IMAGE_SIZE))
            data = np.asarray(image) /255.0
            X=[]
            X.append(data)
            X = np.array(X)


            #1個目のデータの推定結果
            result = model.predict([X])[0]
            #値の大きい方の番号を返す
            predicted = result.argmax()
            #推定確率を出す
            percentage = int(result[predicted] *100)

            #print(self.classes[predicted], percentage)
            return self.classes[predicted], percentage

    def image_src(self):
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()

            return 'data:' + img.file.content_type +";base64," + base64_img
