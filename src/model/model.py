try:
    # Deep learning imports
    from keras.models import Model
    from keras.layers import Input, Activation, Dense
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.merge import Concatenate, Add
    from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam
    from keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error
    from keras.metrics import categorical_accuracy, binary_accuracy
    from keras.preprocessing.image import img_to_array, load_img
    import keras.backend as K
    import tensorflow as tf
    # Python modules
    import os
    import glob
    import random
    from skimage.transform import resize
    import numpy as np
    # Own modules, used for image preprocessing
    from utils import preprocessing as prep
    from utils import processing
except ImportError as err:
    exit("{}: {}".format(__file__, err))


class PredictFace:
    def __init__(self, input_shape=(320, 608, 1), output_shape=(320, 608, 1), batch_size=8, verbose=0):
        """
        Constructor.

        :param input_shape: Input shape (default is 320x608x1)
        :param output_shape: Output shape (default is 320x608x1)
        :param batch_size: Batch size (default is 8)
        """
        self.model = None
        self.built = False
        self.training = False
        self.trained = False
        # Input shape
        self.input_shape = input_shape
        # Output shape
        self.output_shape = output_shape
        # Generator
        self.gan = None
        # Discriminator
        self.d = None
        # Output shape of the discriminator
        self.d_output_shape = None
        # Number of classes (b&w so 1)
        self.n_classes = self.output_shape[-1]
        # Batch size
        self.batch_size = batch_size
        # Verbose
        self.verbose = verbose

    def __del__(self):
        """
        Destructor.

        :return: None
        """
        self.rollback()

    def __generator(self):
        """
        Creates and returns a generator.
        """
        # Inputs
        inputs = Input(self.input_shape, name="generator_input", batch_shape=(self.batch_size,) + self.input_shape)
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
        conc6 = Concatenate(name='conc6')([up_conv5, add4])
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
        conc7 = Concatenate(name='conc7')([up_conv6, add3])
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
        conc8 = Concatenate(name='conc8')([up_conv7, add2])
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
        conv10 = Conv2D(self.n_classes, (1, 1), padding='same', data_format='channels_last', name='conv10_1')(add9)
        acti10 = Activation(tf.nn.sigmoid, name='acti10_1')(conv10)

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

        inputs = Input(self.input_shape[:2] + (self.n_classes * 2,),
                       name="discriminator_input",
                       batch_shape=(self.batch_size*2,)
                                   + self.input_shape[:2]
                                   + (self.n_classes * 2,))

        conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s, s), padding=padding, name="conv2d_error")(inputs)
        conv1 = BatchNormalization(scale=False)(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(conv1)
        conv1 = BatchNormalization(scale=False)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

        conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding)(pool1)
        conv2 = BatchNormalization(scale=False)(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding=padding)(conv2)
        conv2 = BatchNormalization(scale=False)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

        conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding)(pool2)
        conv3 = BatchNormalization(scale=False)(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding)(conv3)
        conv3 = BatchNormalization(scale=False)(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

        conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding)(pool3)
        conv4 = BatchNormalization(scale=False)(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding)(conv4)
        conv4 = BatchNormalization(scale=False)(conv4)
        conv4 = Activation('relu')(conv4)
        pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

        conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding)(pool4)
        conv5 = BatchNormalization(scale=False)(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding)(conv5)
        conv5 = BatchNormalization(scale=False)(conv5)
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

        d.compile(optimizer=Adam(lr=0.0001), loss=d_loss, metrics=['accuracy'])

        return d, d.layers[-1].output_shape[1:]

    def rollback(self):
        # Clear the Keras session
        K.clear_session()
        # Reset parameters
        self.model = None
        self.gan = None
        self.d = None
        self.d_output_shape = None
        self.built = False
        self.training = False
        self.trained = False

    def create_model(self):
        # Instantiate the generator
        self.model = self.__generator()
        # Instantiate the discriminator and retrieve its output shape
        self.d, self.d_output_shape = self.__discriminator()

        # Image input
        img_input = Input(self.input_shape, name="img_input", batch_shape=(self.batch_size,) + self.input_shape)
        # Output input (lol)
        out_input = Input(self.output_shape, name="out_input", batch_shape=(self.batch_size,) + self.input_shape)

        # Model to create the faces
        fake_out = self.model(img_input)
        # Concatenate the output of the model and the image input
        fake_pair = Concatenate()([img_input, fake_out])

        # Create the GAN
        self.gan = Model([img_input, out_input], self.d(fake_pair), name="gan")

        def gan_loss(y_true, y_pred):
            """
            Loss function with binary crossentropy and MSE.

            :param y_true: real image
            :param y_pred: predicted image
            :return: loss
            """
            l_adv = binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
            l_out = mean_squared_error(out_input, fake_out)
            o = tf.transpose(tf.ones(l_out.shape), (2, 1, 0))
            l_adv = tf.transpose(tf.multiply(l_adv, o), (2, 1, 0))
            return tf.add(0.1 * K.cast(l_adv, K.floatx()), K.cast(l_out, K.floatx()))

        # Compile the GAN
        self.gan.compile(optimizer=Adam(lr=0.0001), loss=gan_loss, metrics=[binary_accuracy])

        # Get a summary of the previously created models
        self.model.summary()
        self.d.summary()
        self.gan.summary()

        self.built = True

    def learn(self, epochs=10):
        if not self.built:
            raise ValueError("Create and build the model before using this method (PredictFace.create_model)")

        # Set training to True
        self.training = True
        # Get the folder containing the data
        folder = os.path.join(os.path.realpath(__file__).split("src")[0], "data/CyberextruderUltimate")
        # Verify folder existence
        assert os.path.exists(folder), "{} folder doesn't exist.".format(train_dir)

        nbr_of_imgs = 10205
        train_size = nbr_of_imgs // self.batch_size

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            # Train the discriminator
            data = self.create_data_generator(folder, randomize=True)
            processing.make_trainable(self.d, True)
            for i in range(train_size):
                # Get the next `batch_size` images
                real_imgs, real_preds = next(data)
                # Transform and adapt them to train the discriminator
                d_x_batch, d_y_batch = prep.input2discriminator(real_imgs,
                                                                real_preds,
                                                                self.model.predict(real_imgs,
                                                                                   batch_size=self.batch_size),
                                                                self.d_output_shape)
                # Train the discriminator on this batch
                d_loss, d_acc = self.d.train_on_batch(d_x_batch, d_y_batch)
                if self.verbose >= 1:
                    print("{}/{} -> g_loss = {}, g_acc = {}".format(i + 1, train_size, d_loss, d_acc))

            # Train the GAN
            data = self.create_data_generator(folder, randomize=True)
            processing.make_trainable(self.d, False)
            for i in range(train_size):
                # Get the next `batch_size` images
                real_imgs, real_preds = next(data)
                # Transform and adapt them to train the generator
                g_x_batch, g_y_batch = prep.input2gan(real_imgs,
                                                      real_preds,
                                                      self.d_output_shape)
                # Train the generator on this batch
                g_loss, g_acc = self.gan.train_on_batch(g_x_batch, g_y_batch)
                if self.verbose >= 1:
                    print("{}/{} -> g_loss = {}, g_acc = {}".format(i + 1, train_size, g_loss, g_acc))

        # Set `training` to False (training is over)
        self.training = False
        # Set `trained` to True
        self.trained = True

    def predict(self, data):
        return self.model.predict(data)

    def create_data_generator(self, folder, batch_size=None, randomize=False, training=True):
        """
        Creates a generator which contains a tuple of two lists: one
        for the input images and the other one for the segmented
        (aka labelled) images. The length of these lists is the same
        and is equal to batch_size.

        :param folder: folder in which input and labelled images are.
        :param batch_size: the batch size, must be greater than 1.
        :param randomize: use a random list of data
        :param training: set `training` to True if it is in the training size. Otherwise, set
        `training` to False.
        """
        if not batch_size:
            batch_size = self.batch_size

        if training and batch_size <= 1:
            raise ValueError("During training step, batch size must be greater than 1, not {}.".format(batch_size))

        x_dir = os.path.join(folder, "x")
        y_dir = os.path.join(folder, "y")

        assert os.path.exists(x_dir), "{} doesn't exist.".format(x_dir)
        assert os.path.exists(y_dir), "{} doesn't exist.".format(y_dir)

        xfiles = glob.glob(os.path.join(x_dir, "*.jpg"))
        yfiles = glob.glob(os.path.join(y_dir, "*.jpg"))
        # Number of files
        nbr_files = len(xfiles)

        assert nbr_files == len(yfiles), "Number of x files ({}) is not equal to the number of y files ({}).".format(
            nbr_files,
            len(yfiles)
        )

        # Let's begin the training/validation with the first file
        index = 0
        # Copy
        x_files = xfiles.copy()
        y_files = yfiles.copy()
        while True:
            x, y = list(), list()
            for i in range(batch_size):
                # Get a new index
                if randomize:
                    if not x_files:
                        x_files = xfiles.copy()
                        y_files = yfiles.copy()
                    index = x_files.index(random.choice(x_files))
                else:
                    index = (index + 1) % nbr_files

                # MUST be true (files must have the same name)
                assert os.path.splitext(os.path.basename(x_files[index]))[0] \
                       == os.path.splitext(os.path.basename(y_files[index]))[0], \
                    "({}) is not the labelled image of ({}).".format(
                        y_files[index],
                        x_files[index]
                    )
                # Load images and transform them to arrays
                x_img = img_to_array(load_img(x_files[index], color_mode="grayscale"))
                y_img = img_to_array(load_img(y_files[index], color_mode="grayscale"))
                # Resize images
                x_img = resize(x_img, self.input_shape, preserve_range=True, anti_aliasing=True)
                y_img = resize(y_img, self.input_shape, preserve_range=True, anti_aliasing=True)
                # Change the type to `int`
                x_img = x_img.astype(int)
                y_img = y_img.astype(int)
                # Reduce data values between [0;1]
                x_img = x_img.astype(K.floatx()) / 255.
                y_img = y_img.astype(K.floatx()) / 255.
                # Append these data
                x.append(x_img)
                y.append(y_img)

                if randomize:
                    # Delete these elements
                    del (x_files[index])
                    del (y_files[index])
            yield np.array(x), np.array(y)

    def save(self):
        if not self.built or not self.trained:
            raise ValueError("Build and train this model before saving this model.")

        raise NotImplementedError("Model.save is not implemented.")
