---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---


# LSTM for dummies
Understanding LSTM neural networks from scratch.

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized
dataset = fetch_20newsgroups()

lengths = [len(t) for t in dataset['data']]
print('Samples: {}'.format(len(dataset['target'])))
print('Number of classes: {}'.format(len(np.unique(dataset['target']))))
print('Minimum/maximum number of characters: {}/{}'.format(min(lengths), max(lengths)))

print('Sample:')
print('-'*20)
print(dataset['data'][0].strip())
print('-'*20)
print('Class: '+dataset['target_names'][dataset['target'][0]])
```

<!-- #region -->
## "Vanilla" recurrent neural network
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

At each "time" step $t$, the hidden state $h^{(t)}$ is given by  
$$h^{(t)} = f\left(U x^{(t)} + W h^{(t-1)}\right)$$

where 
* $f$ is the activation function, generally  the hyperbolic tangent (tanh) or the rectified linear unit (ReLU)
* $x^{(t)}$ is the input at time step $t$, for a word sequence $x^{(t)}$ would be the $t$-th word vector representation (either a one-hot-encoding representation, word2vec or GloVe word embedding or an embedding learnt together with the network weights
* $h^{(t-1)}$ is the hidden state at the previous time step
* $U$ and $W$ are weight matrices to be learnt.


Prediction are computed from the hidden state $h^{(t)}$ only: 
$$y^{(t)} = \mathrm{softmax}\left( V h^{(t)} \right)$$ 
where the softmax function $\mathrm{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$ produces probabilities from the real-valued output $V h^{(t)}$.
<!-- #endregion -->

### Custom implementation 

```python
class CustomRNN(object):
     
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = self.initialize_weights(hidden_dim, word_dim)
        self.V = self.initialize_weights(word_dim, hidden_dim)
        self.W = self.initialize_weights(hidden_dim, hidden_dim)
    
    @staticmethod
    def initialize_weights(n_rows, n_cols):
        """ Weights initialization"""
        return np.random.uniform(-np.sqrt(1./n_cols), np.sqrt(1./n_cols), (n_rows, n_cols))
    
    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]
    
    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)
    
    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
 
    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N
    
    
    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])              
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]
    
    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
```

```python
# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) >= 1 and losses[-1][1] >= losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

```

```python
model = RNNNumpy(vocabulary_size)
%timeit model.sgd_step(X_train[10], y_train[10], 0.005)
```

```python
# Prepare design matrix with character indices, padded with zeros 
# at the beginning if the number of characters is lower than maxlen 

maxlen = max(data['anapath'].apply(lambda x: len(x)))
n_samples = len(data)

X = np.zeros((n_samples, maxlen), dtype=np.int64)

for i, text in enumerate(texts):
    for t, char in enumerate(text[-maxlen:]):
        X[i, (maxlen-1-t)] = char_indices[char]

def vec_to_sample_string(v):
    if len(v) < 20:
        return str(v)
    return str(list(v[:5]))[:-1] + ', ..., ' +  str(list(v[-5:]))[1:]

X_train, X_test, y_train, y_test = train_test_split(X, y)
print('First text feature vector: ' + vec_to_sample_string(X_train[0]))
```
