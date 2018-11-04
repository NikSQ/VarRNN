import tensorflow as tf
from src.fc_layer import FCLayer
from src.lstm_cell import LSTMCell

class RNN:
    # TODO: Allow for more flexible fully connected layer
    # TODO: Check if it works correctly for different shapes
    # TODO: Add computation of predictive distribution
    def __init__(self, lstm_shape, fc_shape, gamma):
        self.lstm_cells = []
        self.fc_layer = None
        self.lstm_shape = lstm_shape
        self.fc_shape = fc_shape
        self.gamma = gamma
        self.X = None
        self.Y = None
        self.train_op = None
        self.learning_rate = None
        self.var_free_energy = None
        self.create_network()

    def create_network(self):
        for i in range(len(self.lstm_shape)-1):
            scope = 'lstmlayer' + str(i + 1)
            shape = (self.lstm_shape[i], self.lstm_shape[i+1])
            self.lstm_cells.append(LSTMCell(shape, scope, self.gamma))
        self.fc_layer = FCLayer((self.lstm_shape[len(self.lstm_shape)-1], self.fc_shape[0]), 'fcnlayer', self.gamma)

    # TODO: Include prior in computation of variational free energy
    # TODO: Observe sizes of gradients to determine a good value for clipping
    def create_forward_pass(self, seq_len, beta):
        self.X = tf.placeholder(tf.float32, [None, seq_len, self.lstm_shape[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.fc_shape[0]])
        self.learning_rate = tf.placeholder(tf.float32)

        mu = None
        var = None
        for s in range(seq_len):
            mu = self.X[:, s, :]
            var = tf.fill(tf.shape(mu), 0.)
            for cell in self.lstm_cells:
                if s == 0:
                    mu, var = cell.forward_pass(mu, var, True)
                else:
                    mu, var = cell.forward_pass(mu, var, False)

        exp_log_likelihood = self.fc_layer.forward_pass(mu, var, self.Y, beta)
        self.var_free_energy = -tf.reduce_mean(exp_log_likelihood)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.var_free_energy)

        # TODO: Check why the clipping results in an error
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #gradient = optimizer.compute_gradients(self.var_free_energy)
        #clipped_gradient = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradient]
        #self.train_op = optimizer.apply_gradients(clipped_gradients)

    # TODO: More flexible training (epochs, error feedback, batches)
    # TODO: Add checkpoints for storing variables
    def train(self, X, Y, learning_rate, iterations):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iterations):
                sess.run(self.train_op, feed_dict={self.X: X, self.Y: Y, self.learning_rate: learning_rate})
                print(sess.run(self.var_free_energy, feed_dict={self.X: X, self.Y: Y}))










