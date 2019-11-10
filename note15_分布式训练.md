# TensorFlow 分布式训练 

当我们拥有大量计算资源时，通过使用合适的分布式策略，我们可以充分利用这些计算资源，从而大幅压缩模型训练的时间。针对不同的使用场景，TensorFlow 在 `tf.distribute.Strategy` 中为我们提供了若干种分布式策略，使得我们能够更高效地训练模型。



## 单机多卡训练： `MirroredStrategy`

`tf.distribute.MirroredStrategy` 是一种简单且高性能的，数据并行的同步式分布式策略，主要支持多个 GPU 在同一台主机上训练。使用这种策略时，我们只需实例化一个 `MirroredStrategy` 策略:

```
strategy = tf.distribute.MirroredStrategy()
```

并将模型构建的代码放入 `strategy.scope()` 的上下文环境中:

```
with strategy.scope():
    # 模型构建代码
```

小技巧

可以在参数中指定设备，如:

```
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
```

即指定只使用第 0、1 号 GPU 参与分布式策略。

以下代码展示了使用 `MirroredStrategy` 策略，在 TensorFlow Datasets 中的部分图像数据集上使用 Keras 训练 MobileNetV2 的过程：

```
import tensorflow as tf
import tensorflow_datasets as tfds

num_epochs = 5
batch_size_per_replica = 64
learning_rate = 0.001

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

# 载入数据集并预处理
def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label

# 当as_supervised为True时，返回image和label两个键值
dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(resize).shuffle(1024).batch(batch_size)

with strategy.scope():
    model = tf.keras.applications.MobileNetV2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

model.fit(dataset, epochs=num_epochs)
```

在以下的测试中，我们使用同一台主机上的 4 块 NVIDIA GeForce GTX 1080 Ti 显卡进行单机多卡的模型训练。所有测试的 epoch 数均为 5。使用单机无分布式配置时，虽然机器依然具有 4 块显卡，但程序不使用分布式的设置，直接进行训练，Batch Size 设置为 64。使用单机四卡时，测试总 Batch Size 为 64（分发到单台机器的 Batch Size 为 16）和总 Batch Size 为 256（分发到单台机器的 Batch Size 为 64）两种情况。

| 数据集       | 单机无分布式（Batch Size 为 64） | 单机四卡（总 Batch Size 为 64） | 单机四卡（总 Batch Size 为 256） |
| ------------ | -------------------------------- | ------------------------------- | -------------------------------- |
| cats_vs_dogs | 146s/epoch                       | 39s/epoch                       | 29s/epoch                        |
| tf_flowers   | 22s/epoch                        | 7s/epoch                        | 5s/epoch                         |

可见，使用 MirroredStrategy 后，模型训练的速度有了大幅度的提高。在所有显卡性能接近的情况下，训练时长与显卡的数目接近于反比关系。

`MirroredStrategy` 过程简介

MirroredStrategy 的步骤如下：

- 训练开始前，该策略在所有 N 个计算设备上均各复制一份完整的模型；
- 每次训练传入一个批次的数据时，将数据分成 N 份，分别传入 N 个计算设备（即数据并行）；
- N 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据的梯度；
- 使用分布式计算的 All-reduce 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都有了所有设备的梯度之和；
- 使用梯度求和的结果更新本地变量（镜像变量）；
- 当所有设备均更新本地变量后，进行下一轮训练（即该并行策略是同步的）。

默认情况下，TensorFlow 中的 `MirroredStrategy` 策略使用 NVIDIA NCCL 进行 All-reduce 操作。

为了进一步理解 MirroredStrategy 的过程，以下展示一个手工构建训练流程的示例，相对而言要复杂不少：

\# TODO

## 多机训练： `MultiWorkerMirroredStrategy`

多机训练的方法和单机多卡类似，将 `MirroredStrategy` 更换为适合多机训练的 `MultiWorkerMirroredStrategy` 即可。不过，由于涉及到多台计算机之间的通讯，还需要进行一些额外的设置。具体而言，需要设置环境变量 `TF_CONFIG` ，示例如下:

```
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:20000", "localhost:20001"]
    },
    'task': {'type': 'worker', 'index': 0}
})
```

`TF_CONFIG` 由 `cluster` 和 `task` 两部分组成：

- `cluster` 说明了整个多机集群的结构和每台机器的网络地址（IP + 端口号）。对于每一台机器，`cluster` 的值都是相同的；
- `task` 说明了当前机器的角色。例如， `{'type': 'worker', 'index': 0}` 说明当前机器是 `cluster` 中的第 0 个 worker（即 `localhost:20000` ）。每一台机器的 `task` 值都需要针对当前主机进行分别的设置。

以上内容设置完成后，在所有的机器上逐个运行训练代码即可。先运行的代码在尚未与其他主机连接时会进入监听状态，待整个集群的连接建立完毕后，所有的机器即会同时开始训练。

提示

请在各台机器上均注意防火墙的设置，尤其是需要开放与其他主机通信的端口。如上例的 0 号 worker 需要开放 20000 端口，1 号 worker 需要开放 20001 端口。

以下示例的训练任务与前节相同，只不过迁移到了多机训练环境。假设我们有两台机器，即首先在两台机器上均部署下面的程序，唯一的区别是 `task` 部分，第一台机器设置为 `{'type': 'worker', 'index': 0}` ，第二台机器设置为 `{'type': 'worker', 'index': 1}` 。接下来，在两台机器上依次运行程序，待通讯成功后，即会自动开始训练流程。

```
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json

num_epochs = 5
batch_size_per_replica = 64
learning_rate = 0.001

num_workers = 2
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:20000", "localhost:20001"]
    },
    'task': {'type': 'worker', 'index': 0}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
batch_size = batch_size_per_replica * num_workers

def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label

dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(resize).shuffle(1024).batch(batch_size)

with strategy.scope():
    model = tf.keras.applications.MobileNetV2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

model.fit(dataset, epochs=num_epochs)
```

在以下测试中，我们在 Google Cloud Platform 分别建立两台具有单张 NVIDIA Tesla K80 的虚拟机实例（具体建立方式参见 [后文介绍](https://tf.wiki/zh/appendix/cloud.html#gcp) ），并分别测试在使用一个 GPU 时的训练时长和使用两台虚拟机实例进行分布式训练的训练时长。所有测试的 epoch 数均为 5。使用单机单卡时，Batch Size 设置为 64。使用双机单卡时，测试总 Batch Size 为 64（分发到单台机器的 Batch Size 为 32）和总 Batch Size 为 128（分发到单台机器的 Batch Size 为 64）两种情况。

| 数据集       | 单机单卡（Batch Size 为 64） | 双机单卡（总 Batch Size 为 64） | 双机单卡（总 Batch Size 为 128） |
| ------------ | ---------------------------- | ------------------------------- | -------------------------------- |
| cats_vs_dogs | 1622s                        | 858s                            | 755s                             |
| tf_flowers   | 301s                         | 152s                            | 144s                             |

可见模型训练的速度同样有大幅度的提高。在所有机器性能接近的情况下，训练时长与机器的数目接近于反比关系。