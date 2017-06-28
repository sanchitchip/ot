from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


##convert y_test and y_train to one hot representations in the form of a binary matrix.
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

##  define an input_img tensor for a 32x32 image with 3 channels(RGB).
from keras.layers import Input
input_img = Input(shape = (299, 299, 3))




## Determine proper input shape

input_shape = _obtain_input_shape(input_shape,
default_size=299,
min_size=71,
data_format=K.image_data_format(),
include_top=include_top)

img_input = Input(shape=input_shape)

x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
x = BatchNormalization(name='block1_conv1_bn')(x)
x = Activation('relu', name='block1_conv1_act')(x)
x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
x = BatchNormalization(name='block1_conv2_bn')(x)
x = Activation('relu', name='block1_conv2_act')(x)

residual = Conv2D(128, (1, 1), strides=(2, 2),
padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
x = BatchNormalization(name='block2_sepconv1_bn')(x)
x = Activation('relu', name='block2_sepconv2_act')(x)
x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
x = BatchNormalization(name='block2_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
x = layers.add([x, residual])

residual = Conv2D(256, (1, 1), strides=(2, 2),
padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block3_sepconv1_act')(x)
x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
x = BatchNormalization(name='block3_sepconv1_bn')(x)
x = Activation('relu', name='block3_sepconv2_act')(x)
x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
x = BatchNormalization(name='block3_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
x = layers.add([x, residual])

residual = Conv2D(728, (1, 1), strides=(2, 2),
padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block4_sepconv1_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
x = BatchNormalization(name='block4_sepconv1_bn')(x)
x = Activation('relu', name='block4_sepconv2_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
x = BatchNormalization(name='block4_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
x = layers.add([x, residual])

for i in range(8):
residual = x
prefix = 'block' + str(i + 5)

x = Activation('relu', name=prefix + '_sepconv1_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
x = Activation('relu', name=prefix + '_sepconv2_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
x = Activation('relu', name=prefix + '_sepconv3_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

x = layers.add([x, residual])

residual = Conv2D(1024, (1, 1), strides=(2, 2),
padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block13_sepconv1_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
x = BatchNormalization(name='block13_sepconv1_bn')(x)
x = Activation('relu', name='block13_sepconv2_act')(x)
x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
x = BatchNormalization(name='block13_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
x = layers.add([x, residual])

x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
x = BatchNormalization(name='block14_sepconv1_bn')(x)
x = Activation('relu', name='block14_sepconv1_act')(x)

x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
x = BatchNormalization(name='block14_sepconv2_bn')(x)
x = Activation('relu', name='block14_sepconv2_act')(x)


x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dense(classes, activation='softmax', name='predictions')(x)

inputs = img_input

model = Model(inputs = input_img, outputs = x)
# print model.summary()

##Weight_Path:


## weight loading:
model.load_weights(weights_path)

##compile and fit the model with SGD optimize
from keras.optimizers import SGD
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
