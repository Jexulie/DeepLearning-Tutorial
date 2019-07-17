import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        #! stop before last 2 layers
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            #? adding extra node for bias
            self.W.append(w / np.sqrt(layers[i]))

        #! last 2 layers special case where the input connections need a bias but output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # Shows architecture of network
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    #- Activation Sigmoid
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    #- Activation Sigmoid Deriv
    def sigmoid_deriv(self, x):
        return x * (1 - x)

    #- Fitting
    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # insert a column of 1's as the last entry in the feature matrix
        #? bias trick <--
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    #- Partial Fitting
    def fit_partial(self, x, y):
        #! core of backpropagation...
        A = [np.atleast_2d(x)]

        #? FEEDFORWARD:
        for layer in np.arange(0, len(self.W)):
            #? 'Net Input' to the current layer
            net = A[layer].dot(self.W[layer])


            out = self.sigmoid(net)
            A.append(out)

        #? BACKPROPAGION:
        #* phase 1 - calculate difference between our *prediction* and the true target value
        error = A[-1] - y #! (-1) last entry in the list

        #* apply chain rule and build list of Deltas 'D';
        D = [error * self.sigmoid_deriv(A[-1])]

        #* loop over layers in reverse order ignoring, last 2 since they are already done...
        #* is reverse order becuz we need to work backwards to compute the delta updates for each layer...
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        #* since we looped over our layers in reverse order we need to reverse the deltas...
        #* re-reversing...
        D = D[::-1]

        #* WEIGHT UPDATE PHASE
        for layer in np.arange(0, len(self.W)):
            #* update weigths by taking the dot product of the layer activations with their respective deltas, then multiplying this value by some small learning rate and adding our weight matrix ...
            #* this is where "the ACTUAL LEARING" takes place ...
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    #- Prediction
    def predict(self, X, addBias=True):
        #* init the output prediction as the input features
        p = np.atleast_2d(X)

        #* Bias Check ... ...
        if addBias:
            #* insert 1's Column as the last entry in the feature matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            #* compute output prediction by taking the dot product between current activation value 'p' and the weight matrix associated with the current layer -> pass value through a non-linear activation func...
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p
    
    #- Loss Calculation
    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        
        #* Predicting without Bias
        pred = self.predict(X, addBias=False)
        loss = .5 * np.sum((pred - targets) ** 2)
        
        return loss