#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : resnet.py
#   Author      : YunYang1994
#   Created date: 2019-10-11 19:16:55
#   Description :
#
#================================================================
''' A copy of the above code used to possibly customize the ResNet architecture'''

import tensorflow as tf

class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=strides,
                                            padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                            padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        """
        Adds a shortcut between input and residual block and merges them with "sum"
        """
        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(self.expansion*out_channels, kernel_size=1,
                                           strides=strides, use_bias=False),
                    tf.keras.layers.BatchNormalization()]
                    )
        else:
            self.shortcut = lambda x,_: x
        self.activation = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        # if training: print("=> training network ... ")
        out = self.activation(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x, training)
        return self.activation(out)

class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(out_channels, 1, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, strides, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(out_channels*self.expansion, 1, 1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(self.expansion*out_channels, kernel_size=1,
                                           strides=strides, use_bias=False),
                    tf.keras.layers.BatchNormalization()]
                    )
        else:
            self.shortcut = lambda x,_: x
        self.activation = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        out = self.activation(self.bn1(self.conv1(x), training))
        out = self.activation(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)
        out += self.shortcut(x, training)
        return self.activation(out)

class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = tf.keras.layers.Conv2D(64, 3, 1, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool2d = tf.keras.layers.GlobalAveragePooling2D()
        self.linear = tf.keras.layers.Dense(units=num_classes, activation="sigmoid")
        self.activation = tf.keras.layers.ReLU()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        out = x

        out = self.activation(self.bn1(self.conv1(x), training))
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)

        # For classification
        out = self.avg_pool2d(out)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,14,3], num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)

if __name__ == "__main__":
    from utils import allow_growth
    allow_growth()
    
    model = ResNet18(1024)
    model.build(input_shape=[1, 64, 64, 1])
    
    print(model.summary())
    print(model.predict_on_batch(tf.ones([1, 64, 64, 1], tf.float32)).shape)
