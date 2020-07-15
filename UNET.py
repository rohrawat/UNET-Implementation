import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Sequential

def conv_layer(output_ch, input_shape):
    conv = Sequential()
    conv.add(Conv2D(output_ch,kernel_size=3, activation = 'relu', input_shape = input_shape[1:]))
    conv.add(Conv2D(output_ch, kernel_size=3, activation = 'relu'))
    return conv

def transpose_conv(output_ch):
    trans_conv = Conv2DTranspose(output_ch, kernel_size = 2, strides = 2)
    return trans_conv

def image_crop(input_, output):
    input_size = input_.shape[1]
    output_size = output.shape[1]
    diff = input_size - output_size
    diff = diff // 2
    return input_[:,diff:input_size-diff, diff:input_size-diff,:]

class UNET():
    def __init__(self):
        super(UNET, self).__init__()

        self.maxpool = MaxPooling2D(pool_size = 2)
        self.conv2d = Conv2D(2, kernel_size = 1)
    
    def forward_pass(self, image):
        x0 = conv_layer(64,image.shape)(image)
        x1 = self.maxpool(x0)
        x2 = conv_layer(128, x1.shape)(x1)
        x3 = self.maxpool(x2)
        x4 = conv_layer(256,x3.shape)(x3)
        x5 = self.maxpool(x4)
        x6 = conv_layer(512,x5.shape)(x5)
        x7 = self.maxpool(x6)
        x8 = conv_layer(1024,x7.shape)(x7)
        y = transpose_conv(512)(x8)
        x6_ = image_crop(x6,y)
        x = concatenate([y,x6_],-1)
        y1 = conv_layer(512,x.shape)(x)
        y = transpose_conv(256)(y1)
        x4_ = image_crop(x4,y)
        x = concatenate([y,x4_],-1)
        y1 = conv_layer(256,x.shape)(x)
        y = transpose_conv(128)(y1)
        x2_ = image_crop(x2,y)
        x = concatenate([y,x2_],-1)
        y1 = conv_layer(128,x.shape)(x)
        y = transpose_conv(64)(y1)
        x0_ = image_crop(x0,y)
        x = concatenate([y,x0_],-1)
        y1 = conv_layer(64,x.shape)(x)
        out = self.conv2d(y1)
        print(out.shape)
        


if __name__ == "__main__":
    image = tf.random.normal((1,572,572,3))
    model = UNET()
    model.forward_pass(image)