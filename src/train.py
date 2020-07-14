import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Sequential, Model
from . import Dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def image_crop(input_, output):
    input_size = input_.shape[1]
    output_size = output.shape[1]
    diff = input_size - output_size
    diff = diff // 2
    if (input_size-diff-diff)==output_size:
        return input_[:,diff:input_size-diff,diff:input_size-diff,:]
    else:
        return input_[:,diff:input_size-diff-1,diff:input_size-diff-1,:]

    
def forward_pass(input_size = (256,256,1)):
    image = Input(input_size)
    x0 = Conv2D(64,kernel_size=3, activation = 'relu', padding = 'same')(image)
    x0 = Conv2D(64, kernel_size=3, activation = 'relu', padding = 'same')(x0)
    x1 = MaxPooling2D(pool_size = 2)(x0)
    x2 = Conv2D(128,kernel_size=3, activation = 'relu', padding = 'same')(x1)
    x2 = Conv2D(128, kernel_size=3, activation = 'relu', padding = 'same')(x2)
    x3 = MaxPooling2D(pool_size = 2)(x2)
    x4 = Conv2D(256,kernel_size=3, activation = 'relu', padding = 'same')(x3)
    x4 = Conv2D(256, kernel_size=3, activation = 'relu', padding = 'same')(x4)
    x5 = MaxPooling2D(pool_size = 2)(x4)
    x6 = Conv2D(512,kernel_size=3, activation = 'relu', padding = 'same')(x5)
    x6 = Conv2D(512, kernel_size=3, activation = 'relu', padding = 'same')(x6)
    x7 = MaxPooling2D(pool_size = 2)(x6)
    x8 = Conv2D(1024,kernel_size=3, activation = 'relu', padding = 'same')(x7)
    x8 = Conv2D(1024, kernel_size=3, activation = 'relu', padding = 'same')(x8)


    y = Conv2DTranspose(512, kernel_size = 2, strides = 2, activation = 'relu', padding = 'same')(x8)
    x6_ = image_crop(x6,y)
    x = concatenate([y,x6_], 3)
    print(y.shape)
    print(x6_.shape)
    y1 = Conv2D(512,kernel_size=3, activation = 'relu', padding = 'same')(x)
    y1 = Conv2D(512, kernel_size=3, activation = 'relu', padding = 'same')(y1)
    y = Conv2DTranspose(256, kernel_size = 2, strides = 2,activation = 'relu', padding = 'same')(y1)
    x4_ = image_crop(x4,y)
    print(x4.shape)
    print(x4_.shape)
    print(y.shape)
    x = concatenate([y,x4_],3)
    y1 = Conv2D(256,kernel_size=3, activation = 'relu', padding = 'same')(x)
    y1 = Conv2D(256, kernel_size=3, activation = 'relu', padding = 'same')(y1)
    y = Conv2DTranspose(128, kernel_size = 2, strides = 2, activation = 'relu', padding = 'same')(y1)
    x2_ = image_crop(x2,y)
    x = concatenate([y,x2_],3)
    y1 = Conv2D(128,kernel_size=3, activation = 'relu', padding = 'same')(x)
    y1 = Conv2D(128, kernel_size=3, activation = 'relu', padding = 'same')(y1)
    y = Conv2DTranspose(64, kernel_size = 2, strides = 2, activation = 'relu', padding = 'same')(y1)
    x0_ = image_crop(x0,y)
    x = concatenate([y,x0_],3)
    y1 = Conv2D(64,kernel_size=3, activation = 'relu', padding = 'same')(x)
    y1 = Conv2D(64, kernel_size=3, activation = 'relu', padding = 'same')(y1)
    out = Conv2D(2, kernel_size = 1)(y1)
    out = Conv2D(1, kernel_size = 1, activation = 'sigmoid')(out)
    model = Model(image, out)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model   

if __name__ == "__main__":
    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    image = Dataset.trainGenerator(2,'input/train','image','label',data_gen_args, save_to_dir= None)
    print('done1')
    model = forward_pass()
    print('done2')
    #model_checkpoint = ModelCheckpoint('models/unet.h5', monitor='loss',verbose=1, save_best_only=True)
    print('done3')
    model.fit(image, steps_per_epoch = 1000, epochs = 5)
    model.save_weights('models/unet.h5')
    print('done4')