import tensorflow as tf


class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, initializer='glorot_uniform', **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.initializer = initializer

    def build(self, input_shape):
        # Initialize the (centers) weights and biases of the layer
        self._cw = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer=self.initializer,
                                 trainable=True)
        self._sw = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer=self.initializer,
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

        super(RBFLayer, self).build(input_shape)


    def call(self, inputs):
        # Compute the output of the layer
        # inputs: (batch_size, input_dim)
        # w: (input_dim, units)
        # b: (units,)
        # output: (batch_size, units)
        # diff = tf.expand_dims(inputs, -1) - tf.transpose(self._cw) # (batch_size, input_dim, units)
        # l2_norm = tf.reduce_sum(tf.square(diff), axis=1) # (batch_size, units)
        # output = tf.exp(-self.betas * l2_norm) + self.b # (batch_size, units)

        diff = tf.expand_dims(inputs, -1) - tf.transpose(self._cw)
        squared_diff = tf.square(diff)
        squared_width = tf.square(self._sw)
        squared_distance = tf.reduce_sum(squared_diff / squared_width, axis=1)  # squared distances
        output = tf.exp(-squared_distance)  + self.b #RBF activations
        return output


class RBFNN(tf.keras.Model):
    def __init__(self, rbf_units, output_units=1, hidden_layers:tuple|list=None, learning_rate=0.01, **kwargs):

        super().__init__(**kwargs)
        self.rbfnn_optimizer = tf.keras.optimizers.SGD(learning_rate)
        self._hidden_layers = []
        # Creating the RBF layer and the output layer
        self._rbf_layer = RBFLayer(rbf_units)
        if hidden_layers:
            for layer in hidden_layers:
                self._hidden_layers.append(tf.keras.layers.Dense(layer, activation='relu'))

        self.output_layer = tf.keras.layers.Dense(output_units)
        
        self.output_units = output_units
        self.rbf_units = rbf_units
        self.hidden_layers = hidden_layers

    def call(self, inputs):
        # the output of the model
        # inputs: (batch_size, input_dim)
        # output: (batch_size, output_dim)
        rbf_output = self._rbf_layer(inputs)  # (batch_size, rbf_units)
        for layer in self._hidden_layers:
            rbf_output = layer(rbf_output)  # (batch_size, hidden_units)
        output = self.output_layer(rbf_output)  # (batch_size, output_units)
        return output

    
    def update_weights(self, new_data, target_values):
        with tf.GradientTape() as tape:
            predicted_output = self.call(new_data)
            loss = tf.reduce_mean(tf.square(target_values - predicted_output))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.rbfnn_optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss