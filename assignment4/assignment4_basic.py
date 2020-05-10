import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# exercise 1: vanilla rnn
'''
Preparing Data: Read in the training data, determine the number
of unique characters in the text and set up mapping functions - one
mapping each character to a unique index and another mapping each
index to a character.
'''
def prepareData(path):
    file = open(path, "r")
    book_data = file.read()
    book_data.replace('\n', '') # remove \n because it makes problems

    book_chars = list(set(book_data)) # set of unique characters
    K = len(book_chars)
    char_to_ind =  { val : id for id,val in enumerate(book_chars) } # dict with char and index in set
    ind_to_char =  { id : val for id,val in enumerate(book_chars) }
    return book_data, book_chars, K, char_to_ind, ind_to_char

def seq_of_char_to_onehot(seq_of_char, dim, char_to_ind):
    ind_list =  [char_to_ind[val] for val in list(seq_of_char)]
    one_hot = np.eye(dim)[ind_list].copy()
    one_hot.resize((len(seq_of_char), dim),  refcheck=False)
    return one_hot.T


def onehot_to_seq_of_char(seq_of_oh, ind_to_char ):
    ind_list = np.argmax(seq_of_oh, axis=0)
    char_list = [ind_to_char[ind] for ind in ind_list]
    return char_list


# exercise 2: set hyperparameters
'''
Back-propagation: The forward and the backward pass of the back-
propagation algorithm for a vanilla RNN to eciently compute the
gradients.
AdaGrad updating your RNN's parameters.
Synthesizing text from your RNN: Given a learnt set of parame-
ters for the RNN, a default initial hidden state h0 and an initial input
vector, x0, from which to bootstrap from then you will write a function
to generate a sequence of text.
'''

class RNN():
    def __init__(self, K, book_data, book_chars, char_to_ind, ind_to_char, epsilon=1e-8, m=100, eta=0.1, seq_length=25):
        self.m=m
        self.K = K
        self.eta=eta
        self.seq_length=seq_length
        self.b = np.zeros((m,1))
        self.c = np.zeros((K, 1))
        self.sig = 0.01
        self.U = np.random.randn(m,K) *self.sig
        self.W = np.random.randn(m, m) *self.sig
        self.V = np.random.randn(K, m) *self.sig
        self.b= np.random.randn(m, 1) *self.sig
        self.c = np.random.randn(K,1) *self.sig
        self.book_data = book_data
        self.book_chars = book_chars
        self. char_to_ind = char_to_ind
        self.ind_to_char = ind_to_char
        self.epsilon = epsilon
        self.loss_flag = True

    def synthesize(self, h0, x0, n, char_to_ind, ind_to_char):
        '''
        h0: the hidden state at time 0
        x0: represent the first (dummy) input vector to your RNN
        an integer n denoting the length of the sequence you want to generate.
        '''
        final_out_t = []
        xNext = x0
        h_t = h0

        for t in range(n):
            if t > 0:
                x_t = seq_of_char_to_onehot(xNext, self.K, char_to_ind)
            else:
                x_t = xNext
            a_t, h_t, o_t, p_t = self.evaluate(h_t, x_t)
            label = np.random.choice(self.K, p=p_t[:, 0])  # p: the probabilities associated with each entry in K
            # label = self.sample_a_label(p_t) # old version (did not work as expected)
            xNext = ind_to_char[label]
            final_out_t.append(ind_to_char[label])

        return final_out_t

    def evaluate(self, ht, xt ):
        a_t = np.dot(self.W, ht) + np.dot(self.U, xt) + self.b
        h_t = self.tanh(a_t)
        o_t = np.dot(self.V, h_t) + self.c
        p_t = self.softmax(o_t)
        return a_t, h_t, o_t, p_t

    '''Do not use this. It gives not so good results'''
    def sample_a_label(self, p):
        cp = np.cumsum(p)
        a = np.random.rand()
        cpa = cp-a
        ixs = [j for j in (cpa) if j > 0 ]  # ixs = np.where(cp-a >0)
        ii = np.where(cpa ==ixs[0])
        return ii[0][0]

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        f = np.exp(x - np.max(x))  # avoiding nan for large numbers
        return f / f.sum(axis=0)

    def loss(self, p, y):
        return -np.log(np.dot(y.T, p))

    def forward(self, X, Y, h0):
        loss = 0
        final_out_t = []
        intermediary_p_t = np.zeros((self.K,X.shape[1]))
        intermediary_h_t = np.zeros((self.m,X.shape[1]))
        intermediary_a_t = np.zeros((self.m,X.shape[1]))
        h_t =h0

        for t in range(X.shape[1]):
            a_t, h_t, o_t, p_t = self.evaluate(h_t,  X[:,[t]])
            intermediary_a_t[:,[t]]=a_t
            intermediary_h_t[:,[t]]=h_t
            intermediary_p_t[:,[t]]=p_t
            #final_out_t.append(ind_to_char[self.sample_a_label(p_t)])
            if self.loss_flag:
                loss += self.loss(p_t, Y[:, t])

        return loss, final_out_t, intermediary_a_t, intermediary_h_t, intermediary_p_t

    def compute_gradients(self, A, H, h0, P, X, Y):
        _grad_U = np.zeros(self.U.shape)
        _grad_W = np.zeros(self.W.shape)
        _grad_V = np.zeros(self.V.shape)
        _grad_b = np.zeros(self.b.shape)
        _grad_c = np.zeros(self.c.shape)
        _grad_h_t = np.zeros((self.m, 1))
        _grad_a_t = np.zeros((self.m, 1))
        _grad_A = np.zeros((self.m, Y.shape[1]))

        N = Y.shape[1]

        # grad of loss for cross-enropy and sotfmax layer
        G = -(Y.T - P.T).T
            # grad of loss w.r.t. V
        _grad_V = np.dot(G,H.T)
        _grad_c[:,0] = np.sum(G, axis=1)
        # grad of loss w.r.t. ht
        for t in reversed(range(N)): # iterate backwards
            if t == (N-1):# last h
                _grad_h_t = np.dot(G.T[t], self.V)
            else:
                _grad_h_t = np.dot(G.T[t], self.V) + np.dot(_grad_a_t, self.W)
            _grad_a_t = np.dot(_grad_h_t, np.diag(1 - np.tanh(A[:, t]) ** 2))

            _grad_A[:,t] = _grad_a_t
            _grad_b[:,0] += _grad_a_t

        Hprev= np.delete(np.concatenate((h0,H ), axis=1), -1, -1) # add first column and delete last
        _grad_W = np.dot(_grad_A, Hprev.T)
        _grad_U = np.dot(_grad_A, X.T)

      # clipping
        _grad_U = np.clip(_grad_U, a_min=-5, a_max=5)
        _grad_W = np.clip(_grad_W, a_min=-5, a_max=5)
        _grad_V = np.clip(_grad_V, a_min=-5, a_max=5)
        _grad_b = np.clip(_grad_b, a_min=-5, a_max=5)
        _grad_c = np.clip(_grad_c, a_min=-5, a_max=5)

        return _grad_U, _grad_V, _grad_W, _grad_b, _grad_c

    def computeRelativeError(self, ga, gn, eps):
        return np.absolute(np.subtract(ga, gn)) / np.maximum(np.add(np.absolute(ga), np.absolute(gn)), np.full(ga.shape, eps))

    def compute_gradients_num(self, X, Y, h0,  h = 1e-4):
        grad_b = np.zeros(self.b.shape)
        grad_c = np.zeros(self.c.shape)
        grad_U = np.zeros(self.U.shape)
        grad_V = np.zeros(self.V.shape)
        grad_W = np.zeros(self.W.shape)

      # gradient b
        for i in range(self.b.shape[0]):
            self.b[i] -= h
            c1, _, _, _, _ = self.forward( X, Y, h0)
            self.b[i] += 2*h
            c2, _, _, _, _ = self.forward(X, Y, h0)
            self.b[i] -= h # restore
            grad_b[i] = (c2-c1) / (2*h)

        # gradient c
        for i in range(self.c.shape[0]):
          self.c[i] -= h
          c1, _, _, _, _ = self.forward( X, Y, h0)
          self.c[i] += 2 * h
          c2, _, _, _, _ = self.forward( X, Y, h0)
          self.c[i] -= h  # restore
          grad_c[i] = (c2 - c1) / (2 * h)

        # gradient U
        for i in range(self.U.shape[0]):
          for j in range(self.U.shape[1]):
              self.U[i, j] -= h
              c1, _, _, _, _ = self.forward( X, Y, h0)
              self.U[i, j] += 2 * h
              c2, _, _, _, _ = self.forward( X, Y, h0)
              self.U[i, j] -= h  # restore
              grad_U[i, j]  = (c2 - c1) / (2 * h)

        # gradient V
        for i in range(self.V.shape[0]):
          for j in range(self.V.shape[1]):
              self.V[i, j] -= h
              c1, _, _, _, _ = self.forward( X, Y, h0)
              self.V[i, j] += 2 * h
              c2, _, _, _, _ = self.forward( X, Y, h0)
              self.V[i, j] -= h  # restore
              grad_V[i, j]  = (c2 - c1) / (2 * h)

        # gradient W
        for i in range(self.W.shape[0]):
          for j in range(self.W.shape[1]):
              self.W[i, j] -= h
              c1, _, _, _, _ = self.forward( X, Y, h0)
              self.W[i, j] += 2 * h
              c2, _, _, _, _ = self.forward( X, Y, h0)
              self.W[i, j] -= h  # restore
              grad_W[i, j]  = (c2 - c1) / (2 * h)

        return grad_U, grad_V, grad_W, grad_b, grad_c

    def test_gradients(self, n, book_data):
        h0 = np.zeros((100, 1))
        x_chars = book_data[0:n]
        y_chars = book_data[1:n + 1]

        X = seq_of_char_to_onehot(x_chars, self.K, char_to_ind)
        Y = seq_of_char_to_onehot(y_chars, self.K, char_to_ind)

        loss, final_out_t, intermediary_a_t, intermediary_h_t, intermediary_p_t = rnn.forward(X, Y, h0)

        grad_U, grad_V, grad_W, grad_b, grad_c = self.compute_gradients(intermediary_a_t, intermediary_h_t, h0,
                                                                       intermediary_p_t, X, Y)
        grad_U_num, grad_V_num, grad_W_num, grad_b_num, grad_c_num = self.compute_gradients_num(X, Y, h0)

        eps = 1e-6
        # compute relative errors
        DU = rnn.computeRelativeError(grad_U, grad_U_num, eps)
        DV = rnn.computeRelativeError(grad_V, grad_V_num, eps)
        DW = rnn.computeRelativeError(grad_W, grad_W_num, eps)
        Db = rnn.computeRelativeError(grad_b, grad_b_num, eps)
        Dc = rnn.computeRelativeError(grad_c, grad_c_num, eps)

        print("relative error U ", np.max(DU))
        print("relative error V ", np.max(DV))
        print("relative error W ", np.max(DW))
        print("relative error b ", np.max(Db))
        print("relative error c ", np.max(Dc))

    def train(self, update_steps):
        e =0 # init e with 1
        intermediary_h_t = np.zeros((self.m,self.seq_length))
        mU =  np.zeros(self.U.shape)
        mV =  np.zeros(self.V.shape)
        mW =  np.zeros(self.W.shape)
        mb =  np.zeros(self.b.shape)
        mc =  np.zeros(self.c.shape)
        N = len(self.book_data)
        update_steps_per_epoch = math.ceil((N-1) / self.seq_length)
        num_epochs = math.ceil(update_steps/ update_steps_per_epoch)
        epoch_counter = -1
        output= {new_list: [] for new_list in range(num_epochs)}
        sloss_list = []

        best_loss = math.inf

        print("Running epochs", num_epochs)

        for t in tqdm(range(update_steps)):

            # set hprev
            hprev = intermediary_h_t[:, [-1]]  # hprev should be set to the last computed hidden state by the forward pass in the previous iteration.

            training_sequence = self.book_data[e:e+self.seq_length]
            sequence_labels = self.book_data[e+1:e+self.seq_length+1]
            # increase e
            e += self.seq_length
            # convert to one hot
            X = seq_of_char_to_onehot(training_sequence, self.K, self.char_to_ind)
            Y = seq_of_char_to_onehot(sequence_labels, self.K, self.char_to_ind)

            # forward pass
            # print after 100th update step
            #if t % 100 == 0:
                #self.loss_flag = True
            loss, _, intermediary_a_t, intermediary_h_t, intermediary_p_t = rnn.forward(X, Y, hprev)

            # backward pass
            if intermediary_p_t.shape != Y.shape:
                continue

            grad_U, grad_V, grad_W, grad_b, grad_c = self.compute_gradients(intermediary_a_t, intermediary_h_t, hprev, intermediary_p_t, X, Y)

            # AdaGrad update setp

            mU += np.power(grad_U,2)
            mV += np.power(grad_V,2)
            mW += np.power(grad_W,2)
            mb += np.power(grad_b,2)
            mc += np.power(grad_c,2)

            self.U -= np.multiply(self.eta / np.sqrt(mU+self.epsilon), grad_U)
            self.V -= np.multiply(self.eta / np.sqrt(mV+self.epsilon), grad_V)
            self.W -= np.multiply(self.eta/ np.sqrt(mW+self.epsilon), grad_W)
            self.b -= np.multiply(self.eta/ np.sqrt(mb+self.epsilon), grad_b)
            self.c -= np.multiply(self.eta / np.sqrt(mc+self.epsilon), grad_c)

            # smoothen the loss
            if t == 0:
                smooth_loss = loss
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss

            if t % 1000 == 0:
                print("smooth loss at step ", t, smooth_loss)
                #self.loss_flag= False
                sloss_list.append(smooth_loss)

            if t% 1000 ==0 : # synthesize text
                text = self.synthesize( hprev, X[:,[0]], 200, self.char_to_ind, self.ind_to_char)
                print("".join(text))

            if t % update_steps_per_epoch == 0 or e > (N - self.seq_length - 1):  # reset the iterator
                epoch_counter += 1
                print("\n epoch", epoch_counter)
                e = 0  # init e with 1
                intermediary_h_t = np.zeros((self.m, self.seq_length))
                output[epoch_counter] = self.synthesize(hprev, X[:, [0]], 400, self.char_to_ind, self.ind_to_char)

            # save best parameters
            if loss < best_loss:
                bestU = self.U
                bestV=self.V
                bestW= self.W
                bestb=self.b
                bestc=self.c

        # print results
        self.plot(sloss_list)
        self.print_text(output)

        #setting best parameters
        self.U=bestU
        self.V=bestV
        self.W=bestW
        self.b=bestb
        self.c=bestc

        text = self.synthesize(hprev, X[:, [0]], 1000, self.char_to_ind, self.ind_to_char)
        print("Best parameters:")
        print("".join(text))

    def plot(self, list, save=True):
        plt.plot(list, label="smooth loss")
        plt.xlabel('100 update steps')
        plt.ylabel('smooth loss')
        plt.suptitle('Smooth loss for training')
        plt.legend()
        if save:
            plt.savefig('Result_Pics/sloss.png')
        plt.show()

    def print_text(self, output):
        for k,v in output.items():
            print("epoch", k)
            print("Text: ", "".join(output[k]))



    # main
book_fname = 'Dataset/goblet_book.txt'
book_data, book_chars, K, char_to_ind, ind_to_char = prepareData(book_fname)
oh = seq_of_char_to_onehot(book_data[0:25], K, char_to_ind) # [K x 25]
char_list = onehot_to_seq_of_char(oh, ind_to_char)
oh2 = seq_of_char_to_onehot(char_list, K, char_to_ind)
rnn = RNN(K, book_data, book_chars, char_to_ind, ind_to_char)

'''
  h0: the hidden state at time 0
  x0: represent the first (dummy) input vector to your RNN 
  an integer n denoting the length of the sequence you want to generate.
  '''
#h0 = np.random.randn(100, 1) *rnn.sig
h0 = np.zeros((100, 1))
n=25
x0 = '.'

#rnn.test_gradients(3,book_data)

#loss, xNext, h_t = rnn.synthesize(book_data[0:25], h0, x0, n,char_to_ind, ind_to_char)
rnn.train(100001)


'''
x_chars = book_data[0:rnn.seq_length]
y_chars = book_data[1:rnn.seq_length+1]


X = seq_of_char_to_onehot(x_chars, rnn.K, char_to_ind)
Y = seq_of_char_to_onehot(y_chars, rnn.K, char_to_ind)

loss, final_out_t, intermediary_a_t, intermediary_h_t, intermediary_p_t = rnn.forward(X, Y, h0)


grad_U, grad_V, grad_W, grad_b, grad_c = rnn.compute_gradients( intermediary_a_t, intermediary_h_t, h0, intermediary_p_t, X ,Y )
grad_U_num, grad_V_num, grad_W_num, grad_b_num, grad_c_num = rnn.compute_gradients_num(X ,Y, h0)

eps = 1e-6
# compute relative errors
DU = rnn.computeRelativeError(grad_U, grad_U_num, eps)
DV = rnn.computeRelativeError(grad_V, grad_V_num, eps)
DW = rnn.computeRelativeError(grad_W, grad_W_num, eps)
Db = rnn.computeRelativeError(grad_b, grad_b_num, eps)
Dc = rnn.computeRelativeError(grad_c, grad_c_num, eps)

print("relative error U ", np.max(DU))
print("relative error V ", np.max(DV))
print("relative error W ", np.max(DW))
print("relative error b ", np.max(Db))
print("relative error c ", np.max(Dc))
'''


# continue with 0.4
print("end")