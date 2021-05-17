
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
import numpy as np

def GlobalAvgPool():
    return tf.keras.layers.GlobalAveragePooling2D()

def MaxAvgPool():
    return tf.keras.layers.GlobalMaxPooling2D()


def reshape_into(inputs, input_to_copy):
    return tf.image.resize(inputs, (input_to_copy.shape[1], input_to_copy.shape[2]), method=tf.image.ResizeMethod.BILINEAR)

# convolution
def convolution(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                          dilation_rate=dilation_rate)

# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=True):
    return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0001),
                                  dilation_rate=dilation_rate)


# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=True):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0001),
                                  pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


def max_pool(pool_size=2, stride=2):
    return layers.MaxPool2D(pool_size=(pool_size, pool_size), strides=(stride, stride))

def avg_pool(pool_size=2, stride=2):
    return layers.AveragePooling2D(pool_size=(pool_size, pool_size), strides=(stride, stride))

# convolution
def Conv(n_filters, kernel_size=3, strides=1, dilation_rate=1, use_bias=True):
    return layers.Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=use_bias, 
                         dilation_rate=dilation_rate)

# Traspose convolution
def Conv_trans(n_filters, kernel_size=2, strides=2, dilation_rate=1, use_bias=True):
    return layers.Conv2DTranspose(n_filters, kernel_size, strides=(strides, strides), padding='same', use_bias=use_bias,
                                  kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)
class up_block(tf.keras.Model):
    def __init__(self, n_filters, kernel_size=2, stride=2, dilation_rate=1, trans=True):
        super(up_block, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.conv_trans = Conv_trans(n_filters, kernel_size, stride)
        self.bn = layers.BatchNormalization()


    def call(self, inputs, activation=True, training=True):
        x = self.conv_trans(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)
        return x

class conv_layer(tf.keras.Model):
    def __init__(self, n_filters, kernel_size, stride=1, dilation_rate=1):
        super(conv_layer, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.conv = Conv(self.n_filters, self.kernel_size, self.stride, self.dilation_rate)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, activation=True, normalization=True, training=True):
        x = self.conv(inputs, training=training)
        if normalization:
            x = self.bn(inputs, training=training)
        if activation:
            x = layers.ReLU()(x)
        return x


class conv_block(tf.keras.Model):
    def __init__(self, n_filters, kernel_size=3, stride=1, dilation_rate=1):
        super(conv_block, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.conv1 = Conv(self.n_filters, self.kernel_size, self.stride, self.dilation_rate)
        self.conv2 = Conv(2 * self.n_filters, self.kernel_size, self.stride, self.dilation_rate)
        self.identity = Conv(2 * self.n_filters, self.kernel_size, self.stride, self.dilation_rate)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()



    def call(self, inputs, activation=True, normalization=True, training=True):

        x1 = self.conv1(inputs)
        x1 = self.bn1(x1, training=training)
        x1 = layers.ReLU()(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2, training=training)
        x2 = layers.ReLU()(x2)

        identity = self.identity(inputs)
        # identity = self.bn3(identity, training=training)
        x2 = x2 + identity
        x2 = self.bn3(x2, training=training)
        x2 = layers.ReLU()(x2)
        return x2


class Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = convolution(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, activation=True, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.ReLU()(x)

        return x


# class AdaptivePool(tf.keras.Model):
#     def __init__(self, pool_size, kernel_size=3):
#         super(AdaptivePool, self).__init__()


class DepthMaxPool(tf.keras.Model):
    def __init__(self, pool_size, strides=None, padding="VALID", **kwargs):
        super(DepthMaxPool, self).__init__()
        if strides is None:
            strides = pool_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
    def call(self, inputs, type_='MAX'):
        if type_ == 'MAX':
            return tf.nn.max_pool(inputs,
                              ksize=(1, 1, 1, self.pool_size),
                              strides=(1, 1, 1, self.pool_size),
                              padding=self.padding)
        else:
            return tf.nn.avg_pool(inputs,
                              ksize=(1, 1, 1, self.pool_size),
                              strides=(1, 1, 1, self.pool_size),
                              padding=self.padding)                    

class channel_atttention(tf.keras.Model):
    def __init__(self, n_filters, ratio=16, outputSize=1):
        super(channel_atttention, self).__init__()
        # self.avg_pool = layers.MaxPool2D(pool_size=(1,64,1,1), strides=1)#GlobalAvgPool()
        # self.avg_pool = DepthMaxPool(64)
        # self.max_pool = DepthMaxPool(64)

        # self.avg_pool =  tf.nn.max_pool(s,
                        # ksize=(1, 1, 1, 3),
                        # strides=(1, 1, 1, 3),
                        # padding="VALID")
        # self.max_pool = layers.MaxPool2D(pool_size=(1,64,1,1), strides=1)#MaxAvgPool()
        self.conv1 = Conv(n_filters//ratio, kernel_size=1)
        self.conv2 = Conv(n_filters, kernel_size=1)
        self.outputSize = outputSize
        self.conv3 = Conv(n_filters//ratio, kernel_size=1)
        self.conv4 = Conv(n_filters, kernel_size=1)

    def call(self, inputs, training=True):
        # conv1 = tf.reshape(self.avg_pool(inputs), (1, 1, 1, -1))
        # _, h, w, ch = K.int_shape(inputs)
         
        stride = 256#np.floor(h/self.outputSize).astype(np.int32)
        kernel = 256#h - (self.outputSize-1) * stride
        conv1 = max_pool(pool_size=kernel, stride=stride)(inputs)
        # conv1 = self.avg_pool(inputs)
        # conv1 = tf.reduce_max(inputs, axis=[3], keepdims=True)
        conv1 = self.conv1(conv1, training=training)
        conv1 = layers.ReLU()(conv1)
        conv2 = self.conv2(conv1, training=training)

        # conv3 = tf.reshape(self.max_pool(inputs), (1, 1, 1, -1))
        conv3 = avg_pool(pool_size=kernel, stride=stride)(inputs)
        # conv3 = self.max_pool(inputs)
        # conv3 = tf.reduce_max(inputs, axis=[3], keepdims=True)
        
        conv3 = self.conv3(conv3, training=training)
        conv3 = layers.ReLU()(conv3)
        conv4 = self.conv4(conv3, training=training)
        # layers.Add
        out = conv2 + conv4
        out = tf.keras.activations.sigmoid(out)
        return out

# def adapPool():


def Concat():
    return layers.concatenate()

class Siamese(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, None, 3), n_filters=64, **kwargs):
        super(Siamese, self).__init__(**kwargs)
        print(n_filters, 'Number of filters ')
        self.pool = max_pool()
        # self.concat = Concat()
        # self.conv0_0 = conv_block(n_filters)

        ## Encoder Input 0
        self.conv0_0 = conv_block(n_filters)
        self.conv1_0 = conv_block(2 * n_filters)
        self.conv2_0 = conv_block(4 * n_filters)
        self.conv3_0 = conv_block(8 * n_filters)

        ## Encoder Input 1
        self.conv0_1 = conv_block(n_filters)
        self.conv1_1 = conv_block(2 * n_filters)
        self.conv2_1 = conv_block(4 * n_filters)
        self.conv3_1 = conv_block(8 * n_filters)
        self.conv4_1 = conv_block(16 * n_filters)


        ## Upsampling - Decoder 

        self.conv_up_0_0 = conv_block(n_filters)
        self.conv_up_0_1 = conv_block(n_filters)
        self.conv_up_0_2 = conv_block(n_filters)
        self.conv_up_0_3 = conv_block(n_filters)


        self.conv_up_1_0 = conv_block(2 * n_filters)
        self.conv_up_1_1 = conv_block(2 * n_filters)
        self.conv_up_1_2 = conv_block(2 * n_filters)


        self.conv_up_2_0 = conv_block(4 * n_filters)
        self.conv_up_2_1 = conv_block(4 * n_filters)
        self.conv_up_2_2 = conv_block(4 * n_filters)


        self.conv_up_3_0 = conv_block(8 * n_filters)
        self.conv_up_3_1 = conv_block(8 * n_filters)


        self.up0_1 = up_block(n_filters)
        self.up0_0 = up_block(n_filters)
        self.up0_2 = up_block(n_filters)
        self.up0_3 = up_block(n_filters)


        self.up1_1 = up_block(2 * n_filters)
        self.up1_0 = up_block(2 * n_filters)
        self.up1_2 = up_block(2 * n_filters)


        self.up2_1 = up_block(4 * n_filters)
        self.up2_0 = up_block(4 * n_filters)

        self.up3_1 = up_block(4 * n_filters)
        self.up3_0 = up_block(8 * n_filters)

        self.cam = channel_atttention(4 * n_filters)
        self.cam1 = channel_atttention(n_filters, ratio=4)


        self.final_up0 = up_block(n_filters)
        self.final_up1 = up_block(n_filters)
        self.final_up2 = up_block(n_filters)
        self.final_up3 = up_block(n_filters)


        self.final_conv = Conv(1, kernel_size=1)

    def call(self, inputs, training=True):
        # print(inputs.shape)
        inputs0, inputs1 = inputs[:, :, :256, :], inputs[:, :, 256:, :]
        # inputs0 = Input((256, 512, 3))
        # inputs1 = Input((256, 256, 3))
        x0_0 = max_pool()(self.conv0_0(inputs0, training=training))
        x1_0 = max_pool()(self.conv1_0(x0_0, training=training))
        x2_0 = max_pool()(self.conv2_0(x1_0, training=training))
        x3_0 = max_pool()(self.conv3_0(x2_0, training=training))

        x0_1 = max_pool()(self.conv0_1(inputs1, training=training))
        x1_1 = max_pool()(self.conv1_1(x0_1, training=training))
        x2_1 = max_pool()(self.conv2_1(x1_1, training=training))
        x3_1 = max_pool()(self.conv3_1(x2_1, training=training))
        x4_1 = max_pool()(self.conv4_1(x3_1, training=training))

        concat_x_3 = layers.concatenate([x3_0, x3_1], axis=-1)
        concat_x_2 = layers.concatenate([x2_0, x2_1], axis=-1)
        concat_x_1 = layers.concatenate([x1_0, x1_1], axis=-1)
        concat_x_0 = layers.concatenate([x0_0, x0_1], axis=-1)

        trans_0 = self.up0_0(concat_x_1, training=training)
        _concat_0 = self.conv_up_0_0(layers.concatenate([concat_x_0, trans_0], axis=-1), training=training)

        trans_1 = self.up1_0(concat_x_2, training=training)
        _concat_1 = self.conv_up_1_0(layers.concatenate([trans_1, concat_x_1], axis=-1), training=training)
        trans_1_1 = self.up0_1(_concat_1, training=training)
        _concat_1_1 = self.conv_up_0_1(layers.concatenate([trans_1_1, concat_x_0, _concat_0], axis=-1), training=training)

        trans_2 = self.up2_0(concat_x_3, training=training)
        _concat_2 = self.conv_up_2_0(layers.concatenate([trans_2, concat_x_2], axis=-1), training=training)
        trans_2_1 = self.up1_1(_concat_2, training=training)
        _concat_2_1 = self.conv_up_1_1(layers.concatenate([trans_2_1, concat_x_1, _concat_1], axis=-1), training=training)
        trans_2_2 = self.up0_2(_concat_2_1, training=training)
        _concat_2_2 = self.conv_up_0_2(layers.concatenate([trans_2_2, concat_x_0, _concat_1_1, _concat_0], axis=-1), training=training)

        trans_3 = self.up3_0(x4_1, training=training)
        _concat_3 = self.conv_up_3_0(layers.concatenate([trans_3, concat_x_3], axis=-1), training=training)
        trans_3_1 = self.up2_1(_concat_3, training=training)
        _concat_3_1 = self.conv_up_2_1(layers.concatenate([trans_3_1, concat_x_2, _concat_2], axis=-1), training=training)
        trans_3_2 = self.up1_2(_concat_3_1, training=training)
        _concat_3_2 = self.conv_up_1_2(layers.concatenate([trans_3_2, concat_x_1, _concat_2_1, _concat_1], axis=-1), training=training)     
        trans_3_3 = self.up0_3(_concat_3_2, training=training)
        _concat_3_3 = self.conv_up_0_3(layers.concatenate([trans_3_3, concat_x_0, _concat_0, _concat_2_2, _concat_1_1], axis=-1), training=training)     


        _concat_3_3 = self.final_up3(_concat_3_3, training=training)
        _concat_2_2 = self.final_up2(_concat_2_2, training=training)
        _concat_1_1 = self.final_up1(_concat_1_1, training=training)
        _concat_0 = self.final_up0(_concat_0, training=training)

        out = layers.concatenate([_concat_0, _concat_1_1, _concat_2_2, _concat_3_3], axis=-1)

        add_out = _concat_0 + _concat_1_1 + _concat_2_2 + _concat_3_3

        CAM = self.cam(out, training=training)
        CAM1 =self.cam1(add_out, training=training)
        CAM1 = layers.concatenate([CAM1, CAM1, CAM1, CAM1], axis=-1)
        # CAM1 = tf.repeat(CAM1, (1, 1, 1, 4))

        add_cam = out + CAM1
        out = layers.Multiply()([CAM, add_cam])
        # out = CAM * add_cam
        out = self.final_conv(out, training=training)

        # # x = reshape_into(out, inputs)

        # out = tf.keras.activations.softmax(out, axis=-1)
        out = tf.keras.activations.sigmoid(out)
        # model = tf.keras.Model(inputs0, inputs1, out)
        return out

class ResNet50Seg(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(ResNet50Seg, self).__init__(**kwargs)
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights=weights,
                                                             input_shape=input_shape, pooling='avg')

        output_2 = base_model.get_layer('conv2_block2_out').output
        output_3 = base_model.get_layer('conv3_block3_out').output
        output_4 = base_model.get_layer('conv4_block5_out').output
        output_5 = base_model.get_layer('conv5_block3_out').output
        outputs = [output_5, output_4, output_3, output_2]

        self.model_output = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        self.conv_up1 = Conv_BN(1024, 3)
        self.conv_up2 = Conv_BN(512, 3)
        self.conv_up3 = Conv_BN(256, 3)

        self.classify = convolution(num_classes, 1, strides=1, dilation_rate=1, use_bias=True)

    def call(self, inputs, training=True):

        outputs = self.model_output(inputs, training=training)

        x = reshape_into(outputs[0], outputs[1])
        x = self.conv_up1(x, training=training) + outputs[1]
        x = reshape_into(x, outputs[2])


        x = self.conv_up2(x, training=training) + outputs[2]
        x = reshape_into(x, outputs[3])


        x = self.conv_up3(x, training=training) + outputs[3]
        x = self.classify(x, training=training)

        x = reshape_into(x, inputs)

        x = tf.keras.activations.softmax(x, axis=-1)

        return x


 

