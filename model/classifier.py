import numpy as np # linear algebra
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, Lambda, MaxPooling2D, BatchNormalization, Activation, add, advanced_activations, Dense, Flatten, GlobalAveragePooling2D, Dropout

#from keras import backend as K
from keras.layers.merge import concatenate
from keras.initializers import glorot_normal, glorot_uniform, he_normal, he_uniform

relu = 'relu'
selu = 'selu'
cls_num = 10
between_cls_num = 4
psk_cls_num, qam_cls_num, fsk_cls_num = 3,3,3
def bottleneck_block(inp, kernel_size=(1,4), filters=[16,16,32], name='bottleneck_block'):
    out = Conv2D(filters[0], kernel_size=(1,1), strides=(1,1), init='he_normal', 
                 name=name+'_conv_1', padding='same', activation=None)(inp)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)

    out = Conv2D(filters[1], kernel_size=kernel_size, strides=(1,1), init='he_normal', 
                 name=name+'_conv_2', padding='same', activation=None)(out)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    
    out = Conv2D(filters[2], kernel_size=(1,1), strides=(1,1), init='he_normal', 
                 name=name+'_conv_3', padding='same', activation=None)(out)
    out = BatchNormalization()(out)    
    
    
    out = add([out, inp])
    out = Activation(relu)(out)
    return out    
def vgg():
    inp = Input([2, 1024, 1]) # 1024
    out = Conv2D(64, kernel_size=(1,4), padding='same', init='he_normal', activation=None)(inp)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 512
    
    out = Conv2D(64, kernel_size=(1,4), padding='same',init='he_normal', activation=None)(out)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 256
    
    out = Conv2D(64, kernel_size=(2,1), padding='valid',init='he_normal', activation=None)(out)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 128
    
    out = Conv2D(64, kernel_size=(1,4), padding='same',init='he_normal', activation=None)(out)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 64
    
    out = Conv2D(64, kernel_size=(1,4), padding='same',init='he_normal', activation=None)(out)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 32
    
    out = Conv2D(64, kernel_size=(1,4), padding='same',init='he_normal', activation=None)(out)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 16
    
    out = Conv2D(64, kernel_size=(1,4), padding='same',init='he_normal', activation=None)(out)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 8       
    
    out = Flatten()(out)
    out = Dense(128, activation=selu, init='he_normal')(out)
    out = Dense(128, activation=selu, init='he_normal')(out)
    out = Dense(cls_num, activation='softmax', init='he_normal')(out)
    return Model(inputs=inp, outputs=out)


def resnet_26():
    inp = Input([None, None, 1]) # 1024
    # stage 0
    out = Conv2D(64, (1,4), padding='same', init='he_normal', activation=None, name='s0')(inp)
    out = BatchNormalization()(out)
    out = Activation(relu)(out) # 1024
    
    # stage 1
    out = bottleneck_block(out, kernel_size=(1,4), filters=[16,16,64], name='s1_block_1')
    out = bottleneck_block(out, kernel_size=(1,4), filters=[16,16,64], name='s1_block_2')
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 512
    # stage 2
    out = bottleneck_block(out, kernel_size=(1,4), filters=[16,16,64], name='s2_block_1')
    out = bottleneck_block(out, kernel_size=(1,4), filters=[16,16,64], name='s2_block_2')
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 256
    # stage 3
    out = Conv2D(64, (2,3), padding='valid', init='he_normal', activation=None, name='s3')(out)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 127
    # stage 4
    out = bottleneck_block(out, kernel_size=(1,4), filters=[16,16,64], name='s4_block_1')
    out = bottleneck_block(out, kernel_size=(1,4), filters=[16,16,64], name='s4_block_2')
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 64
    # stage 5
    out = bottleneck_block(out, kernel_size=(1,4), filters=[16,16,64], name='s5_block_1')
    out = bottleneck_block(out, kernel_size=(1,4), filters=[16,16,64], name='s5_block_2')
    out = MaxPooling2D(pool_size=(1,2), padding='same')(out) # 32
    
    # stage 6
    out = GlobalAveragePooling2D()(out)
    out = Dense(cls_num, init='he_normal', activation='softmax', name='output')(out)
    return Model(inputs=inp, outputs=out)

#m = resnet_()
#m.summary()

dr = 0.3
def cnn2():
    inp = Input([2, 1024, 1])
    out = Conv2D(64, (1,4), padding='same', activation=None, init='he_normal', name='c1')(inp)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    out = MaxPooling2D((1,2), padding='same', name='mp1')(out)
    
    out = Conv2D(64, (2,3), padding='valid', activation=None, init='he_normal', name='c2')(out)
    out = BatchNormalization()(out)
    out = Activation(relu)(out)
    out = Flatten()(out)
    
    out = Dropout(dr)(out)
    out = Dense(256, activation=selu, init='he_normal', name='fc1')(out)
    out = Dropout(dr)(out)
    out = Dense(cls_num, activation='softmax', init='he_normal', name='output')(out)
    return Model(inputs=inp, outputs=out)
#m = cnn2()
#m.summary()