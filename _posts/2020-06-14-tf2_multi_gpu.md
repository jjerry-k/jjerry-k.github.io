---
layout: post
title: TensorFlow Multi GPU 사용법
category: [DeepLearning]
tags: [TensorFlow]
sitemap :
changefreq : daily
---

 이번 포스팅은 Multi GPU 시스템에서 Google의 머신러닝 오픈 소스 플랫폼인 [TensorFlow](https://www.tensorflow.org/)사용법에 관한 것입니다!  
거두절미하고 바로 코딩으로 들어가겠습니다!  



## Single GPU 예시
- 다음 코드는 Single GPU를 이용하여 mnist data를 분류하는 코드입니다. 
``` python
# Import Package
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, datasets, utils

# Data Prepare
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
train_x, test_x = np.expand_dims(train_x/255., -1), np.expand_dims(test_x/255., -1)
print("Train Data's Shape : ", train_x.shape, train_y.shape)
print("Test Data's Shape : ", test_x.shape, test_y.shape)

# Build Network
cnn = models.Sequential()
cnn.add(layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1,)))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Conv2D(32, 3, activation='relu'))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Flatten())
cnn.add(layers.Dense(10, activation='softmax'))

cnn.compile(optimizer=optimizers.Adam(), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])                
print("Network Built!")

# Training Network
epochs=10
batch_size = 4096
history = cnn.fit(train_x, train_y, epochs=10, batch_size=batch_size, validation_data=(test_x, test_y))
```

## Multi GPU 예시
- 다음 코드는 Multi GPU를 이용한 코드입니다. 
``` python
# Import Package
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, datasets, utils

# Data Prepare
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
train_x, test_x = np.expand_dims(train_x/255., -1), np.expand_dims(test_x/255., -1)
print("Train Data's Shape : ", train_x.shape, train_y.shape)
print("Test Data's Shape : ", test_x.shape, test_y.shape)

# Build Network
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1,)))
    cnn.add(layers.MaxPool2D())
    cnn.add(layers.Conv2D(32, 3, activation='relu'))
    cnn.add(layers.MaxPool2D())
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(10, activation='softmax'))

    cnn.compile(optimizer=optimizers.Adam(), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])                
print("Network Built!")

# Training Network
epochs=10
batch_size_each_gpu = 4096
batch_size = batch_size_each_gpu*len(gpus)

history = cnn.fit(train_x, train_y, epochs=10, batch_size=batch_size, validation_data=(test_x, test_y))
```

어렵지 않습니다. `Build Network` 주석 부분과 `Training Network` 부분에 `batch_size`만 조금 수정해주시면 끝납니다!  
![img](https://jjerry-k.github.io/public/img/tf_multi_gpu/bob.png)  

하지만 이렇게 하면 무식하게 GPU의 모든 메모리를 할당합니다.  
그렇기 떄문에 다음과 같이 코드를 추가하여 `필요한 만큼` 할당하도록 합니다. 

## 필요한 만큼의 GPU 메모리만 사용하기
``` python
# Import Package
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, datasets, utils

# Data Prepare
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
train_x, test_x = np.expand_dims(train_x/255., -1), np.expand_dims(test_x/255., -1)
print("Train Data's Shape : ", train_x.shape, train_y.shape)
print("Test Data's Shape : ", test_x.shape, test_y.shape)

# For Efficiency
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Build Network
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1,)))
    cnn.add(layers.MaxPool2D())
    cnn.add(layers.Conv2D(32, 3, activation='relu'))
    cnn.add(layers.MaxPool2D())
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(10, activation='softmax'))

    cnn.compile(optimizer=optimizers.Adam(), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])                
print("Network Built!")

# Training Network
epochs=10
batch_size_each_gpu = 4096
batch_size = batch_size_each_gpu*len(gpus)

history = cnn.fit(train_x, train_y, epochs=10, batch_size=batch_size, validation_data=(test_x, test_y))
```

기존 Multi GPU 코드와 달라진 점은 
1. `For Efficiency`라는 부분이 추가.
2. `strategy = tf.distribute.MirroredStrategy()`을 `Build Network`에서 `For Efficiency`의 가장 첫번째 라인으로 이동.

이렇게 변경 후 실행 후 nvidia-smi와 같은 모니터링 툴을 확인해보시면 이전과는 다르게 GPU 메모리를 필요한 만큼만 사용하는걸 보실 수 있습니다!



## P.S 
- 다음은 뭘로 포스팅하지...