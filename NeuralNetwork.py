# Authors: Cooper Cox & Anuk Centellas
# Description: Program 1 for DATA 471, creating an arbitarily deep neural network
# Date: 10/28/24

import argparse
import numpy as np

# defining activation functions and their derivatives
def sigmoid(x) :
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x) :
    bbb_one = np.ones(x.shape)
    return np.multiply(sigmoid(x), (np.subtract(bbb_one, sigmoid(x))))

def tanh_prime(x) :
    bbb_one = np.ones(x.shape)
    return np.subtract(bbb_one, np.square(np.tanh(x)))

def ReLU(x) :
    return np.maximum(0, x)

def ReLU_prime(x) :
    # 1 if greater than or equal to 0, 0 if less than 0
    x = np.where(x >= 0, 1, 0)
    return x

# softmax function for classification
def softmax(x) :
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

# shuffle function to shuffle feature data and target data in the same way (so they line up correctly)
def shuffle(feats, targets) :
    p = np.random.permutation(feats.shape[0])
    return feats[p], targets[p]

# functions to initialize lists W and B with matrices of the correct dimensions
def initialize_W() :
    W_initial = np.empty((D, L))
    W_layers = np.empty((l - 1, L, L))
    W_final = np.empty((L, C))
    W = [W_initial]
    for i in range (len(W_layers)) :
        W.append(W_layers[i])
    W.append(W_final)
    return W

def initialize_B() :
    b_layers = np.empty((l, L, 1))
    b_final = np.empty((C, 1))
    B = []
    for i in range (len(b_layers)) :
        B.append(b_layers[i])
    B.append(b_final)
    return B

# function to populate W and B with random numbers within the init_range
def populate_W(W) :
    W[0] = np.random.uniform(low=-args.INIT_RANGE, high=args.INIT_RANGE, size=(D,L))
    for i in range(1, l) :
        W[i] = np.random.uniform(low=-args.INIT_RANGE, high=args.INIT_RANGE, size=(L,L))
    W[l] = np.random.uniform(low=-args.INIT_RANGE, high=args.INIT_RANGE, size=(L,C))
    return W

def populate_B(B) :
    for i in range(l) :
        B[i] = np.random.uniform(low=-args.INIT_RANGE, high=args.INIT_RANGE, size=(L, 1))
    B[l] = np.random.uniform(low=-args.INIT_RANGE, high=args.INIT_RANGE, size=(C, 1))
    return B

# functions to initialize the A, Z, and deltas lists with matices of the correct dimensions
def initialize_A(A_0) :
    A_layers = np.empty((l, L, mb))
    A_final = np.empty((C, mb))
    A = [A_0]
    for i in range (len(A_layers)) :
        A.append(A_layers[i])
    A.append(A_final)
    return A

def initialize_Z(size) :
    Z = []
    Z_layers = np.empty((l, L, size))
    Z_final = np.empty((C, size))
    for i in range (len(Z_layers)) :
        Z.append(Z_layers[i])
    Z.append(Z_final)
    return Z

def initialize_deltas() :
    deltas = []
    delta_layers = np.empty((l, L, mb))
    delta_last = np.empty((C, mb))
    for i in range(len(delta_layers)) :
        deltas.append(delta_layers)
    deltas.append(delta_last)
    return deltas

# main training loop
def train() :
    # initialize and populate W, B, Z, W gradients, and B gradients
    W = initialize_W()
    B = initialize_B()
    W = populate_W(W)
    B = populate_B(B)
    Z = initialize_Z(mb)
    W_grads = initialize_W()
    B_grads = initialize_B()
    num_updates = 0
    epoch = 0

    while num_updates < args.TOTAL_UPDATES :

        # shuffle the data
        train_feat_shuffled, train_target_shuffled = shuffle(train_feat_data, train_target_data)
        dev_feat_shuffled, dev_target_shuffled = shuffle(dev_feat_data, dev_target_data)

        # for loop runs for one epoch
        for batch in range(0, int(N/mb)) :
            # taking a minibatch of the data
            mb_x = np.transpose(train_feat_shuffled[(batch * mb) % (N + 1):((batch + 1) * mb) % (N + 1)])
            mb_y = np.transpose(train_target_shuffled[(batch * mb):((batch + 1) * mb)])
            x_dev = np.transpose(dev_feat_shuffled)
            y_dev = np.transpose(dev_target_shuffled)
            A = initialize_A(mb_x)

            if args.VERBOSE_DIR != None :
                write_params_file(num_updates, W, B)

            # printing out info before any updating has happened
            if num_updates == 0 :
                A_dev = initialize_A(x_dev)
                Z_dev = initialize_Z(np.transpose(x_dev).shape[0])
                dev_losses = evaluate(A_dev, W, B, Z_dev, y_dev, True)
                if args.VERBOSE_DIR != None :
                    train_mb_losses = evaluate(A, W, B, Z, mb_y, False)
                    print("Epoch", f"{epoch:04}", "UPDATE", f"{num_updates:06}:", "minibatch=" + str(round(train_mb_losses, 3)), "dev=" + str(round(dev_losses, 3)))
                else :
                    print("Epoch", f"{epoch:04}", "UPDATE", f"{num_updates:06}:", "dev=" + str(round(dev_losses, 3)))
                num_updates += 1
            

            # performing feed forward and getting the final predictions
            A, Z = feedforward(A, W, B, Z)
            predictions = A[l + 1]
            predictions = np.transpose(predictions)

            if (args.PROBLEM_MODE == "C") :
                # softmaxing predictions for classification data
                for i in range(predictions.shape[0]) :
                    predictions[i] = softmax(predictions[i])

            # performing back prop and updating the weights and biases
            W_grads, B_grads = backprop(predictions, mb_y, A, W, Z, W_grads, B_grads)
            W, B = update_weights(W, B, W_grads, B_grads)

            # printing epoch, update, and evaluation info
            if num_updates % args.REPORT_FREQ == 0 :
                A_dev = initialize_A(x_dev)
                Z_dev = initialize_Z(np.transpose(x_dev).shape[0])
                dev_losses = evaluate(A_dev, W, B, Z_dev, y_dev, True)
                if args.VERBOSE_DIR != None :
                    train_mb_losses = evaluate(A, W, B, Z, mb_y, False)
                    print("Epoch", f"{epoch:04}", "UPDATE", f"{num_updates:06}:", "minibatch=" + str(round(train_mb_losses, 3)), "dev=" + str(round(dev_losses, 3)))
                else :
                    print("Epoch", f"{epoch:04}", "UPDATE", f"{num_updates:06}:", "dev=" + str(round(dev_losses, 3)))

            if args.VERBOSE_DIR != None :
                write_grad_file(num_updates, W_grads, B_grads)
                write_mb_file(num_updates, mb_x, mb_y)
            
            num_updates += 1

            if num_updates > args.TOTAL_UPDATES :
                break

        epoch += 1

    if args.VERBOSE_DIR != None:
        write_params_file(num_updates, W, B)

# functions to write parameters, minibatch, and gradients to files for verbose mode
def write_params_file(num_updates, W, B) :
    if num_updates != 0:
        num_updates = num_updates-1
    params_filename = args.VERBOSE_DIR + "/params_" + f"{num_updates:06}" + ".npz"
    W_at = dict()
    B_at = dict()
    for i in range(len(W)) :
        W_at["W_" + str(i + 1)] = np.transpose(W[i].astype(np.float32))
        B_at["b_" + str(i + 1)] = np.transpose(B[i].astype(np.float32))
    np.savez(params_filename, **W_at, **B_at)
    
def write_mb_file(num_updates, mb_x, mb_y) :
    if args.PROBLEM_MODE == 'C' :
        mb_y = one_hot(mb_y)
    mb_x = np.transpose(mb_x)
    minibatch_filename = args.VERBOSE_DIR + "/minibatch_" + f"{num_updates:06}" + ".npz"
    np.savez(minibatch_filename, FEATURES=mb_x.astype(np.float32), TARGETS=mb_y.astype(np.float32))

def write_grad_file(num_updates, W_grads, B_grads) :
    grads_filename = args.VERBOSE_DIR + "/gradients_" + f"{num_updates:06}" + ".npz"
    W_grads_at = dict()
    B_grads_at = dict()
    for i in range(len(W_grads)) :
        W_grads_at["grad_W_" + str(i + 1)] = np.transpose(W_grads[i].astype(np.float32))
        B_grads_at["grad_b_" + str(i + 1)] = np.transpose(B_grads[i].astype(np.float32))
    np.savez(grads_filename, **W_grads_at, **B_grads_at)
    
# helper function to calculate the Z and A for each iteration of feed forward
def calculate_Z_A(A, W, b, layer) :
    Z = np.add(np.matmul(np.transpose(W), A), b)
    if layer == l and args.PROBLEM_MODE != 'C':  
        return Z, Z 
    if args.HIDDEN_UNIT_ACTIVATION == 'sig' :
        A = sigmoid(Z)
    elif args.HIDDEN_UNIT_ACTIVATION == 'relu' :
        A = ReLU(Z)
    else:
        A = np.tanh(Z)
    return A, Z

# feed forward function, fills in A and Z matrices
def feedforward(A, W, B, Z) :
    for layer in range(l + 1) :
        A[layer + 1], Z[layer] = calculate_Z_A(A[layer], W[layer], B[layer], layer)
    return A, Z

# used for creating a one hot encoding of the true outputs for classification
def one_hot(x) :
    x = x.astype(int)
    one_hot = np.zeros((x.size, C), dtype=int)
    one_hot[np.arange(x.size), x] = 1
    return one_hot

# helper function to calculate partial derivatives
def partials(a, b) :
    partial = np.divide(np.matmul(a, np.transpose(b)), mb)
    return partial

# helper function to claculate delta based on the Z, W, and previous delta
def delta_calc(Z, W, delta) :
    if args.HIDDEN_UNIT_ACTIVATION == 'sig' :
        f_prime = sigmoid_prime(Z)
    elif args.HIDDEN_UNIT_ACTIVATION == 'relu' :
        f_prime = ReLU_prime(Z)
    else:
        f_prime = tanh_prime(Z)
    prev_delta = np.multiply(f_prime, np.matmul(W, delta))
    return prev_delta

# backprop function, calculates the W gradients, B gradients, and deltas at each iteration
def backprop(predictions, mb_y, A, W, Z, W_grads, B_grads) :
    if args.PROBLEM_MODE == 'C' :
        mb_y = one_hot(mb_y)
    else :
        mb_y = np.transpose(mb_y)
    deltas = initialize_deltas()
    #black board bold one
    bbb_one = np.ones((1, mb))
    deltas[l] = np.transpose(np.subtract(predictions, mb_y))
    for i in range(len(A) - 1, 0, -1) :
        W_grads[i - 1] = partials(A[i - 1], deltas[i - 1])
        B_grads[i - 1] = np.transpose(partials(bbb_one, deltas[i - 1]))
        if i == 1 :
            break
        deltas[i - 2] = delta_calc(Z[i - 2], W[i - 1], deltas[i - 1])
    return W_grads, B_grads

# function for updating the weights based on the gradients and learn rate
def update_weights(W, B, W_grads, B_grads) :
    W_updates = W
    B_updates = B
    for i in range(len(W)) :
        W_updates[i] = np.subtract(W_updates[i], (args.LEARNRATE * W_grads[i]))
        B_updates[i] = np.subtract(B_updates[i], (args.LEARNRATE * B_grads[i]))
    return W_updates, B_updates

# function to convert softmaxed predictions into predictions of a single class for each input
def argmax_predictions(predictions) :
    # transposing so softmaxing works properly
    predictions = np.transpose(predictions)
    for i in range(predictions.shape[0]) :
        predictions[i] = softmax(predictions[i])
    predictions = np.transpose(predictions)
    class_pred = np.argmax(predictions, axis=0)
    # reshaping to a np matrix instead of a vector
    class_pred = np.reshape(class_pred, (class_pred.shape[0], 1))
    class_pred = np.transpose(class_pred)
    return class_pred

# functions used for evaluating on the dev set and the minibatch (minibatch is for verbose mode)
def evaluate(A, W, B, Z, Y, dev) :
    if args.PROBLEM_MODE == 'C' :
        if dev == True :
            predictions, Z = feedforward(A, W, B, Z)
            predictions = predictions[l + 1]
        # verbose mode
        else :
            predictions = A[l + 1]
        class_pred = argmax_predictions(predictions)
        correct = 0
        for i in range(len(class_pred[0])) :
            if class_pred[0][i] == Y[0][i] :
                correct += 1
        total_correct = (correct/len(class_pred[0]))    
        return total_correct
    else :
        if dev == True :
            predictions, Z = feedforward(A, W, B, Z)
            predictions = predictions[l + 1]
        # verbose mode
        else :
            predictions = A[l + 1]
            breakpoint()
        mse_arr = (np.square(predictions - Y)).mean(axis=1)
        # other option for mse_arr, both work
        # mse_arr = (np.square(np.transpose(predictions) - np.transpose(Y))).mean(axis=0)
        mse = mse_arr[0]
        return mse

if __name__ == '__main__' :
    
    # create a parser and parse through command line arguments, stored in args
    parser = argparse.ArgumentParser(description="process command line args.")
    parser.add_argument('-train_feat', '--TRAIN_FEAT_FN', type=str)
    parser.add_argument('-train_target', '--TRAIN_TARGET_FN', type=str)
    parser.add_argument('-dev_feat', '--DEV_FEAT_FN', type=str)
    parser.add_argument('-dev_target', '--DEV_TARGET_FN', type=str)
    parser.add_argument('-nunits', '--NUM_HIDDEN_UNITS', type=int)
    parser.add_argument('-nlayers', '--NUM_HIDDEN_LAYERS', type=int)
    parser.add_argument('-hidden_act', '--HIDDEN_UNIT_ACTIVATION', type=str)
    parser.add_argument('-type', '--PROBLEM_MODE', type=str)
    parser.add_argument('-output_dim', '--OUTPUT_DIM', type=int)
    parser.add_argument('-total_updates', '--TOTAL_UPDATES', type=int)
    parser.add_argument('-learnrate', '--LEARNRATE', type=float)
    parser.add_argument('-init_range', '--INIT_RANGE', type=float)
    parser.add_argument('-mb', '--MINIBATCH_SIZE', type=int)
    parser.add_argument('-report_freq', '--REPORT_FREQ', type=int)
    parser.add_argument('-v', '--VERBOSE_DIR', type=str)
    args = parser.parse_args()

    # load data
    train_feat_data = np.loadtxt(args.TRAIN_FEAT_FN)
    train_target_data = np.loadtxt(args.TRAIN_TARGET_FN)
    dev_feat_data = np.loadtxt(args.DEV_FEAT_FN)
    dev_target_data = np.loadtxt(args.DEV_TARGET_FN)

    # get dimensions and other important numbers
    N = train_feat_data.shape[0]
    D = train_feat_data.shape[1]
    L = args.NUM_HIDDEN_UNITS
    l = args.NUM_HIDDEN_LAYERS
    if args.MINIBATCH_SIZE > N :
        mb = N
    else :
        mb = args.MINIBATCH_SIZE
    C = args.OUTPUT_DIM

    # reshape data into np matrices instead of vectors
    if args.PROBLEM_MODE == "C" :
        train_target_data = np.reshape(train_target_data, (train_target_data.shape[0], 1))
        dev_target_data = np.reshape(dev_target_data, (dev_target_data.shape[0], 1))
    else :
        train_target_data = np.reshape(train_target_data, (train_target_data.shape[0], C))
        dev_target_data = np.reshape(dev_target_data, (dev_target_data.shape[0], C))

    # run the training loop
    train()