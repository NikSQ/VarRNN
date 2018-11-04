from src.rnn import RNN
import numpy as np

lstm_shape = (1, 7, 2)
fc_shape = [1]
seq_len = 10
rnn = RNN(lstm_shape, fc_shape, 10)
rnn.create_forward_pass(seq_len, 1)

X = np.arange(0, 10, 0.01)
Y = np.square(X)
items = 5
X_tr = np.zeros((items, seq_len, 1))
Y_tr = np.zeros((items, 1))
for i in range(items):
    X_tr[i, :, 0] = X[i:(i+seq_len)]
    Y_tr[i] = Y[i+1]

# TODO: Learning results in NaN of the variational free energy... check the problem
# Probably caused by tf.log in computation expected log-likelihood
rnn.train(X_tr, Y_tr, 0.01, 20000)
