# https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/3-Neural_Network_Architecture/resnet.py

import tensorflow as tf


class BasicBlock_Transposed(tf.keras.Model):
    expansion = 1

    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock_Transposed, self).__init__()

        self.conv1 = tf.keras.layers.Conv2DTranspose(out_channels, kernel_size=3, strides=strides, use_bias=False)

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2DTranspose(out_channels, kernel_size=3, strides=1, use_bias=False)

        self.bn2 = tf.keras.layers.BatchNormalization()

        """
        Adds a shortcut between input and residual block and merges them with "sum"
        """
        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(self.expansion * out_channels, kernel_size=1,
                                                strides=strides, use_bias=False),
                tf.keras.layers.BatchNormalization()]
            )
        else:
            self.shortcut = lambda x, _: x

        def call(self, x, training=False):
            # if training: print("=> training network ... ")
            out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
            out = self.bn2(self.conv2(out), training=training)
            out += self.shortcut(x, training)
            return tf.nn.relu(out)


class Bottleneck_Transposed(tf.keras.Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1):
        super(Bottleneck_Transposed, self).__init__()

        self.conv1 = tf.keras.layers.Conv2DTranspose(out_channels, 1, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2DTranspose(out_channels, 3, strides, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2DTranspose(out_channels * self.expansion, 1, 1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(self.expansion * out_channels, kernel_size=1,
                                                strides=strides, use_bias=False),
                tf.keras.layers.BatchNormalization()]
            )
        else:
            self.shortcut = lambda x, _: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)
        out += self.shortcut(x, training)
        return tf.nn.relu(out)


class ResNet_Decoder(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = tf.keras.layers.Conv2DTranspose(64, 3, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.pool = tf.keras.layers.UpSampling2D((4, 4), interpolation="nearest")

        self.linear = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        out = self.pool(self.bn1(self.conv1(x)))
        out = tf.nn.relu(out, training)
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)

        # For classification

        out = tf.reshape(out, (out.shape[0], -1))
        out = self.linear(out)
        return out


def ResNet18_Decoder():
    return ResNet_Decoder(BasicBlock_Transposed, [2, 2, 2, 2])


def ResNet34_Decoder():
    return ResNet_Decoder(BasicBlock_Transposed, [3, 4, 6, 3])


def ResNet50_Decoder():
    return ResNet_Decoder(Bottleneck_Transposed, [3, 4, 14, 3])


def ResNet101_Decoder():
    return ResNet_Decoder(Bottleneck_Transposed, [3, 4, 23, 3])


def ResNet152_Decoder():
    return ResNet_Decoder(Bottleneck_Transposed, [3, 8, 36, 3])

