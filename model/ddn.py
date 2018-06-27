import numpy as np # linear algebra
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, Lambda, MaxPooling2D, BatchNormalization, Activation, add, advanced_activations, Dense

#from keras import backend as K
from keras.layers.merge import concatenate
from keras.initializers import glorot_normal, glorot_uniform, he_normal, he_uniform

act_func = 'relu'

def conv_block(inp, kernel_size=(1,4), filters=[16,16], name='conv_block'):
    out = Conv2D(filters[0], kernel_size=kernel_size, strides=(1,1), init='he_normal', 
                 name=name+'_conv_1', padding='same', activation=None)(inp)
    out = BatchNormalization()(out)
    out = Activation(act_func)(out)

    out = Conv2D(filters[1], kernel_size=kernel_size, strides=(1,1), init='he_normal', 
                 name=name+'_conv_2', padding='same', activation=None)(out)
    out = BatchNormalization()(out)
    
    out = add([out, inp])
    out = Activation(act_func)(out)
    return out
    
def bottleneck_block(inp, kernel_size=(1,4), filters=[16,16,64], name='bottleneck_block'):
    out = Conv2D(filters[0], kernel_size=(1,1), strides=(1,1), init='he_normal', 
                 name=name+'_conv_1', padding='same', activation=None)(inp)
    out = BatchNormalization()(out)
    out = Activation(act_func)(out)

    out = Conv2D(filters[1], kernel_size=kernel_size, strides=(1,1), init='he_normal', 
                 name=name+'_conv_2', padding='same', activation=None)(out)
    out = BatchNormalization()(out)
    out = Activation(act_func)(out)
    
    out = Conv2D(filters[2], kernel_size=(1,1), strides=(1,1), init='he_normal', 
                 name=name+'_conv_3', padding='same', activation=None)(out)
    out = BatchNormalization()(out)    
    
    
    out = add([out, inp])
    out = Activation(act_func)(out)
    return out    


    
def resnet():
    inp = Input([None, None, 1])
    out = Conv2D(16, (1,4), strides=(1,1), init='he_normal', 
                 name='input_layer', padding='same', activation=None)(inp)
    out = BatchNormalization()(out)
    out = Activation(act_func)(out)
    
    out = conv_block(out, name='cblock_1')
    out = conv_block(out, name='cblock_2')
    out = conv_block(out, name='cblock_3')
    out = conv_block(out, name='cblock_4')
    out = conv_block(out, name='cblock_5')
    out = conv_block(out, name='cblock_6')
    out = conv_block(out, name='cblock_7')
    out = conv_block(out, name='cblock_8')
    out = conv_block(out, name='cblock_9')
    out = conv_block(out, name='cblock_10')
    
    out = Conv2D(1, (1,1), strides=(1,1), init='he_normal', 
                 name='output_layer', padding='same', activation=None)(out)
    out = BatchNormalization()(out)
    
    out = add([out, inp])
    return Model(inputs=inp, outputs=out)

def resnet_buttleneck():
    inp = Input([None, None, 1])
    out = Conv2D(64, (1,4), strides=(1,1), init='he_normal', 
                 name='input_layer', padding='same', activation=None)(inp)
    out = BatchNormalization()(out)
    out = Activation(act_func)(out)

    out = bottleneck_block(out, name='bblock_1')
    out = bottleneck_block(out, name='bblock_2')
    out = bottleneck_block(out, name='bblock_3')
    out = bottleneck_block(out, name='bblock_4')
    out = bottleneck_block(out, name='bblock_5')
    #out = bottleneck_block(out, name='bblock_6')
    #out = bottleneck_block(out, name='bblock_7')
    #out = bottleneck_block(out, name='bblock_8')
    #out = bottleneck_block(out, name='bblock_9')
    #out = bottleneck_block(out, name='bblock_10')

    out = Conv2D(1, (1,1), strides=(1,1), init='he_normal', 
                 name='output_layer', padding='same', activation=None)(out)
    out = BatchNormalization()(out)

    out = add([out, inp])
    return Model(inputs=inp, outputs=out)    

#md = resnet_buttleneck()
#md.summary()

def side_out(x, factor):
    # ensemble all feature maps
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)
    
    # recover the original size
    kernel_size = (1, factor)
    x = Conv2DTranspose(1, kernel_size=(1,factor), strides=(1, factor), padding='same',
                        use_bias=False, activation=None, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    return x

def u_net_side_fuse():
    inp = Input([2, None, 1])
    x = Conv2D(64, (1,4), strides=(1,1), init='he_normal', 
                 name='input_layer', padding='same', activation=None)(inp)
    x = BatchNormalization()(x)
    x = Activation(act_func)(x)
    # x: 2, 1024, 64
    
    # stage A: down sampling
    s1 = bottleneck_block(x, kernel_size=(1,4), filters=[16,16,64], name='s1_block')
    s2 = MaxPooling2D(pool_size=(1,2), padding='same')(s1)
    # s2: 2, 512, 64
    
    s2 = bottleneck_block(s2, kernel_size=(1,4), filters=[16,16,64], name='s2_block')
    s3 = MaxPooling2D(pool_size=(1,2), padding='same')(s2)
    # s3: 2, 256, 64
    
    s3 = bottleneck_block(s3, kernel_size=(1,4), filters=[16,16,64], name='s3_block')
    
    o3 = side_out(s3, factor=4)
    # o3: 2, 1024, 1
    
    # stage B: up-sampling
    u2 = Conv2DTranspose(64, kernel_size=(1,2), strides=(1,2), padding='same', name='up_2')(s3) # u2: 2, 512, 64 
    u2 = concatenate([u2, s2], name='concat_2') # u2: 2, 512, 128?
    u2 = bottleneck_block(u2, kernel_size=(1,4), filters=[16,16,128], name='u2_block')
    o2 = side_out(u2, factor=2)
    
    u1 = Conv2DTranspose(64, kernel_size=(1,2), strides=(1,2), padding='same', name='up_1')(u2) # u1: 2, 1024, 64 
    u1 = concatenate([u1, s1], name='concat_1') # u1: 2, 1024, 128?
    u1 = bottleneck_block(u1, kernel_size=(1,4), filters=[16,16,128], name='u1_block')
    o1 = side_out(u1, factor=1)
    
    # stage C: fuse
    fuse = concatenate([o3, o2, o1], name='concat_fuse', axis=-1)
    fuse = Conv2D(1, (1,1), padding='same', activation=None, name='conv_fuse')(fuse)
    o1 = add([o1, inp], name='o1')
    o2 = add([o2, inp], name='o2')
    o3 = add([o3, inp], name='o3')
    o_fuse = add([fuse, inp], name='o_fuse')
    
    model = Model(inputs=[inp], outputs=[o1, o2, o3, o_fuse])
    return model
    
    
#m = u_net_side_fuse()
#m.summary()