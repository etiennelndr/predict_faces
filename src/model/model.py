try:
    from keras.models import Model
    from keras.layers import Input
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.merge import Concatenate
    from keras.layers.pooling import GlobalAveragePooling2D
    from keras.losses import binary_crossentropy, categorical_crossentropy
    from keras.metrics import categorical_accuracy, binary_accuracy

    from utils import preprocessing as prep
except ImportError as err:
    exit(err)


class PredictFace:
    def __init__(self, input_shape=(320, 600, 3)):
        self.model = None
        self.built = False
        self.training = False
        self.trained = False
        # Input shape
        self.input_shape = input_shape
        # Generator
        self.gan = None
        # Discriminator
        self.d = None
        # Output shape of the discriminator
        self.d_output_shape = None

    def __del__(self):
        pass

    def __generator(self):
        """
        Creates and returns a generator.
        """
        # Inputs
        inputs = Input(self.input_shape)
        # ----- First Convolution - Down-convolution -----
        # 5x5 Convolution
        conv1 = Conv2D(8, (5, 5), padding='same', data_format='channels_last', name='conv1_1')(inputs)
        acti1 = Activation(tf.nn.relu, name='acti1')(conv1)
        # Down-convolution
        down_conv1 = Conv2D(16, (2, 2), strides=(2, 2), data_format='channels_last', name='down_conv1_1')(acti1)

        # ----- Second Convolution - Down-convolution -----
        # 5x5 Convolution
        conv2 = Conv2D(16, (5, 5), padding='same', data_format='channels_last', name='conv2_1')(down_conv1)
        acti2 = Activation(tf.nn.relu, name='acti2_1')(conv2)
        # 5x5 Convolution
        conv2 = Conv2D(16, (5, 5), padding='same', data_format='channels_last', name='conv2_2')(acti2)
        acti2 = Activation(tf.nn.relu, name='acti2_2')(conv2)
        # Add layer
        add2 = Add(name='add2_1')([down_conv1, acti2])
        # Down-convolution
        down_conv2 = Conv2D(32, (2, 2), strides=(2, 2), data_format='channels_last', name='down_conv2_1')(add2)

        # ----- Third Convolution - Down-convolution -----
        # 5x5 Convolution
        conv3 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv3_1')(down_conv2)
        acti3 = Activation(tf.nn.relu, name='acti3_1')(conv3)
        # 5x5 Convolution
        conv3 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv3_2')(acti3)
        acti3 = Activation(tf.nn.relu, name='acti3_2')(conv3)
        # 5x5 Convolution
        conv3 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv3_3')(acti3)
        acti3 = Activation(tf.nn.relu, name='acti3_3')(conv3)
        # Add layer
        add3 = Add(name='add3_1')([down_conv2, acti3])
        # Down-convolution
        down_conv3 = Conv2D(64, (2, 2), strides=(2, 2), data_format='channels_last', name='down_conv3_1')(add3)

        # ----- Fourth Convolution - Down-convolution -----
        # 5x5 Convolution
        conv4 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv4_1')(down_conv3)
        acti4 = Activation(tf.nn.relu, name='acti4_1')(conv4)
        # 5x5 Convolution
        conv4 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv4_2')(acti4)
        acti4 = Activation(tf.nn.relu, name='acti4_2')(conv4)
        # 5x5 Convolution
        conv4 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv4_3')(acti4)
        acti4 = Activation(tf.nn.relu, name='acti4_3')(conv4)
        # Add layer
        add4 = Add(name='add4_1')([down_conv3, acti4])
        # Down-convolution
        down_conv4 = Conv2D(128, (2, 2), strides=(2, 2), data_format='channels_last', name='down_conv4_1')(add4)

        # ----- Fifth Convolution -----
        # 5x5 Convolution
        conv5 = Conv2D(128, (5, 5), padding='same', data_format='channels_last', name='conv5_1')(down_conv4)
        acti5 = Activation(tf.nn.relu, name='acti5_1')(conv5)
        # 5x5 Convolution
        conv5 = Conv2D(128, (5, 5), padding='same', data_format='channels_last', name='conv5_2')(acti5)
        acti5 = Activation(tf.nn.relu, name='acti5_2')(conv5)
        # 5x5 Convolution
        conv5 = Conv2D(128, (5, 5), padding='same', data_format='channels_last', name='conv5_3')(acti5)
        acti5 = Activation(tf.nn.relu, name='acti5_3')(conv5)
        # Add layer
        add5 = Add(name='add5_1')([down_conv4, acti5])
        # Up-convolution
        up_conv5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), data_format='channels_last', name='up_conv5')(add5)

        # ----- Sixth Convolution -----
        # Concatenation
        conc6 = Concatenate(axis=3, name='conc6')([up_conv5, add4])
        # 5x5 Convolution
        conv6 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv6_1')(conc6)
        acti6 = Activation(tf.nn.relu, name='acti6_1')(conv6)
        # 5x5 Convolution
        conv6 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv6_2')(acti6)
        acti6 = Activation(tf.nn.relu, name='acti6_2')(conv6)
        # 5x5 Convolution
        conv6 = Conv2D(64, (5, 5), padding='same', data_format='channels_last', name='conv6_3')(acti6)
        acti6 = Activation(tf.nn.relu, name='acti6_3')(conv6)
        # Add layer
        add6 = Add(name='add6_1')([up_conv5, acti6])
        # Up-convolution
        up_conv6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), data_format='channels_last', name='up_conv6')(add6)

        # ----- Seventh Convolution -----
        # Concatenation
        conc7 = Concatenate(axis=3, name='conc7')([up_conv6, add3])
        # 5x5 Convolution
        conv7 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv7_1')(conc7)
        acti7 = Activation(tf.nn.relu, name='acti7_1')(conv7)
        # 5x5 Convolution
        conv7 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv7_2')(acti7)
        acti7 = Activation(tf.nn.relu, name='acti7_2')(conv7)
        # 5x5 Convolution
        conv7 = Conv2D(32, (5, 5), padding='same', data_format='channels_last', name='conv7_3')(acti7)
        acti7 = Activation(tf.nn.relu, name='acti7_3')(conv7)
        # Add layer
        add7 = Add(name='add7_1')([up_conv6, acti7])
        # Up-convolution
        up_conv7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), data_format='channels_last', name='up_conv7')(add7)

        # ----- Eighth Convolution -----
        # Concatenation
        conc8 = Concatenate(axis=3, name='conc8')([up_conv7, add2])
        # 5x5 Convolution
        conv8 = Conv2D(16, (5, 5), padding='same', data_format='channels_last', name='conv8_1')(conc8)
        acti8 = Activation(tf.nn.relu, name='acti8_1')(conv8)
        # 5x5 Convolution
        conv8 = Conv2D(16, (5, 5), padding='same', data_format='channels_last', name='conv8_2')(acti8)
        acti8 = Activation(tf.nn.relu, name='acti8_2')(conv8)
        # Add layer
        add8 = Add(name='add8_1')([up_conv7, acti8])
        # Up-convolution
        up_conv8 = Conv2DTranspose(8, (2, 2), strides=(2, 2), data_format='channels_last', name='up_conv8')(add8)

        # ----- Ninth Convolution -----
        # 5x5 Convolution
        conv9 = Conv2D(8, (5, 5), padding='same', data_format='channels_last', name='conv9_1')(up_conv8)
        acti9 = Activation(tf.nn.relu, name='acti9_1')(conv9)
        # Add layer
        add9 = Add(name='add9_1')([up_conv8, acti9])

        # ----- Tenth Convolution -----
        conv10 = Conv2D(3, (1, 1), padding='same', data_format='channels_last', name='conv10_1')(add9)
        acti10 = Activation(tf.nn.softmax, name='acti10_1')(conv10)

        # Create the generator model
        g = Model(inputs, acti10, name="generator")

        return g

    def __discriminator(self):
        """
        Creates and returns a discriminator.
        """
        n_filters = 8
        k = 3
        s = 2
        padding = 'same'

        inputs = Input(self.input_shape[:2] + (self.n_classes + 3,), name="discriminator_input")

        conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s, s), padding=padding)(inputs)
        conv1 = BatchNormalization(scale=False, axis=3)(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(conv1)
        conv1 = BatchNormalization(scale=False, axis=3)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

        conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding)(pool1)
        conv2 = BatchNormalization(scale=False, axis=3)(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding=padding)(conv2)
        conv2 = BatchNormalization(scale=False, axis=3)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

        conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding)(pool2)
        conv3 = BatchNormalization(scale=False, axis=3)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding)(conv3)
        conv3 = BatchNormalization(scale=False, axis=3)(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

        conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding)(pool3)
        conv4 = BatchNormalization(scale=False, axis=3)(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding)(conv4)
        conv4 = BatchNormalization(scale=False, axis=3)(conv4)
        conv4 = Activation('relu')(conv4)
        pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

        conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding)(pool4)
        conv5 = BatchNormalization(scale=False, axis=3)(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding)(conv5)
        conv5 = BatchNormalization(scale=False, axis=3)(conv5)
        conv5 = Activation('relu')(conv5)

        gap = GlobalAveragePooling2D()(conv5)
        outputs = Dense(1, activation='sigmoid')(gap)

        d = Model(inputs, outputs, name="discriminator")

        def d_loss(y_true, y_pred):
            """
            TODO
            :param y_true:
            :param y_pred:
            :return:
            """
            return binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))

        d.compile(optimizer=Adam(lr=0.0001, beta_1=0.5), loss=d_loss, metrics=['accuracy'])

        return d, d.layers[-1].output_shape[1:]

    def create_model(self):
        # Instantiate the generator
        self.model = self.__generator()
        # Instantiate the discriminator and retrieve its output shape
        self.d, self.d_output_shape = self.__discriminator()

        # Image input
        img_input = Input(self.input_shape, name="img_input")
        # Output input (lol)
        out_input = Input(self.input_shape, name="out_input")

        # Model to create the faces
        fake_out = self._model(img_input)
        # Concatenate the creation model and the image input
        fake_pair = Concatenate(axis=3)([img_input, fake_out])

        # Create the GAN
        self.gan = Model([img_input, out_input], self.d(fake_pair), name="gan")

        def gan_loss(y_true, y_pred):
            """
            TODO
            :param y_true:
            :param y_pred:
            :return:
            """
            l_adv = binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
            l_out = categorical_crossentropy(K.batch_flatten(out_input), K.batch_flatten(fake_out))
            return 0.1 * l_adv + l_out

        # Compile the GAN
        self.gan.compile(optimizer=Adam(lr=0.0001), loss=gan_loss, metrics=[binary_accuracy, categorical_accuracy])

        # Get a summary of the previously created models
        self.get_model().summary()
        self.d.summary()
        self.gan.summary()

        self.built = True

    def learn(self):
        self.training = True

        self.training = False
        self.trained = True

    def predict(self):
        return None
