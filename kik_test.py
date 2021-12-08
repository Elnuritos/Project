import numpy as np
import tensorflow as tf



tf.compat.v1.disable_eager_execution()
crypto_msg_len = N = 16
crypto_key_len = 16
batch_size = 512
epochs = 50
learning_rate = 0.0008 #Optimizer learning rate

# Function to generate n random messages and keys
def gen_data(n=batch_size, msg_len=crypto_msg_len, key_len=crypto_key_len):
    #return (np.random.randint(0, 2, size=(n, msg_len))*2-1), (np.random.randint(0, 2, size=(n, key_len))*2-1)
    return (np.random.randint(0, 2, size=(n, msg_len))), (np.random.randint(0, 2, size=(n, key_len)))

def conv1d(input_, filter_shape, stride, name="conv1d"):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('w', shape=filter_shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        conv = tf.nn.conv1d(input=input_, filters=w, stride=stride, padding='SAME')
        return conv
# Placeholder variables for Message and Key
msg = tf.compat.v1.placeholder("float", [None, crypto_msg_len])
key = tf.compat.v1.placeholder("float", [None, crypto_key_len])

# Weights for fully connected layers
w_alice = tf.compat.v1.get_variable("alice_w", shape=[2 * N, 2 * N], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
w_bob = tf.compat.v1.get_variable("bob_w", shape=[2 * N, 2 * N], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
w_eve1 = tf.compat.v1.get_variable("eve_w1", shape=[N, 2 * N], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
w_eve2 = tf.compat.v1.get_variable("eve_w2", shape=[2 * N, 2 * N], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

# Alice's Machine Network
# FC layer -> Conv Layer (4 1-D convolutions)
alice_input = tf.concat(axis=1, values=[msg, key])
alice_hidden = tf.nn.sigmoid(tf.matmul(alice_input, w_alice))
alice_hidden = tf.expand_dims(alice_hidden, 2)

h0 = tf.nn.relu(conv1d(alice_hidden, [4,1,2], stride=1, name="alice"+'_h0_conv'))
h1 = tf.nn.relu(conv1d(h0, [2,2,4], stride=2, name="alice"+'_h1_conv'))
h2 = tf.nn.relu(conv1d(h1, [1,4,4], stride=1, name="alice"+'_h2_conv'))
h3 = tf.nn.tanh(conv1d(h2, [1,4,1], stride=1, name="alice"+'_h3_conv'))
alice_output = tf.squeeze(h3) # eliminate dimensions of size 1 from the shape of a tensor

# Bob's Machine Network (gets the output (cipher text) of Alice's network)
# FC layer -> Conv Layer (4 1-D convolutions)
bob_input = tf.concat(axis=1, values=[alice_output, key])
bob_hidden = tf.nn.sigmoid(tf.matmul(bob_input, w_bob))
bob_hidden = tf.expand_dims(bob_hidden, 2)

h0 = tf.nn.relu(conv1d(bob_hidden, [4,1,2], stride=1, name="bob"+'_h0_conv'))
h1 = tf.nn.relu(conv1d(h0, [2,2,4], stride=2, name="bob"+'_h1_conv'))
h2 = tf.nn.relu(conv1d(h1, [1,4,4], stride=1, name="bob"+'_h2_conv'))
h3 = tf.nn.tanh(conv1d(h2, [1,4,1], stride=1, name="bob"+'_h3_conv'))
bob_output = tf.squeeze(h3) # eliminate dimensions of size 1 from the shape of a tensor

# Eve's Machine Network
# FC layer -> FC layer -> Conv Layer (4 1-D convolutions)
eve_input = alice_output
eve_hidden1 = tf.nn.sigmoid(tf.matmul(eve_input, w_eve1))
eve_hidden2 = tf.nn.sigmoid(tf.matmul(eve_hidden1, w_eve2))
eve_hidden2 = tf.expand_dims(eve_hidden2, 2)

h0 = tf.nn.relu(conv1d(eve_hidden2, [4,1,2], stride=1, name="eve"+'_h0_conv'))
h1 = tf.nn.relu(conv1d(h0, [2,2,4], stride=2, name="eve"+'_h1_conv'))
h2 = tf.nn.relu(conv1d(h1, [1,4,4], stride=1, name="eve"+'_h2_conv'))
h3 = tf.nn.tanh(conv1d(h2, [1,4,1], stride=1, name="eve"+'_h3_conv'))
eve_output = tf.squeeze(h3)
alice_errors, bob_errors, eve_errors = [], [], []

# Loss Functions
decrypt_err_eve = tf.reduce_mean(input_tensor=tf.abs(msg - eve_output))
decrypt_err_alice = tf.reduce_mean(input_tensor=tf.abs(msg - alice_output))
loss_alice = decrypt_err_alice + (1. - decrypt_err_eve) ** 2.
decrypt_err_bob = tf.reduce_mean(input_tensor=tf.abs(msg - bob_output))
loss_bob = decrypt_err_bob + (1. - decrypt_err_eve) ** 2.

# Get training variables corresponding to each network
t_vars = tf.compat.v1.trainable_variables()
alice_vars = [var for var in t_vars if 'alice_' in var.name]
bob_vars =   [var for var in t_vars if 'bob_' in var.name]
eve_vars =   [var for var in t_vars if 'eve_' in var.name]

# Build the optimizers, can play with different optimizers

'''
alice_optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1, 
                       use_locking=False, name='Adagrad').minimize(loss_alice, var_list=alice_vars)
bob_optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1, 
                        use_locking=False, name='Adagrad').minimize(loss_bob, var_list=bob_vars)
eve_optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1, 
                        use_locking=False, name='Adagrad').minimize(decrypt_err_eve, var_list=eve_vars)

alice_optimizer = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss_alice, var_list=alice_vars)   
bob_optimizer = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss_bob, var_list=bob_vars)
eve_optimizer = tf.train.MomentumOptimizer(0.01, 0.9).minimize(decrypt_err_eve, var_list=eve_vars)

'''
alice_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_alice, var_list=alice_vars)   
bob_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_bob, var_list=bob_vars)
eve_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(decrypt_err_eve, var_list=eve_vars)

def train(sess):
    # Begin Training
    tf.compat.v1.global_variables_initializer().run(session=sess)
    for i in range(epochs):
        iterations = 2000

        alice_loss, _, _ = _train('alice', iterations, sess)
        alice_errors.append(alice_loss)
        if(i==0 or i==epochs-1):
            print( 'Training Alice, Epoch:', i + 1 , ' error: ', alice_loss)

        _, bob_loss, _ = _train('bob', iterations, sess)
        bob_errors.append(bob_loss)
        if(i==0 or i==epochs-1):
            print( 'Training Bob, Epoch:', i + 1, ' error: ', bob_loss)

        _, _, eve_loss = _train('eve', iterations, sess)
        eve_errors.append(eve_loss) 
        if(i==0 or i==epochs-1):
            print( 'Training Eve, Epoch:', i + 1, ' error: ', eve_loss)


def _train(network, iterations, sess):
    alice_decrypt_error, bob_decrypt_error, eve_decrypt_error = 1., 1., 1.

    bs = batch_size
    # Train Eve for two minibatches to give it a slight computational edge
    if network == 'eve':
        bs *= 2

    for i in range(iterations):
        msg_in_val, key_val = gen_data(n=bs, msg_len=crypto_msg_len, key_len=crypto_key_len)
        feed_dict={msg: msg_in_val, key: key_val}
        if network == 'alice':
            _, decrypt_err = sess.run([alice_optimizer, decrypt_err_alice], feed_dict = feed_dict)
            alice_decrypt_error = min(alice_decrypt_error, decrypt_err)
        elif network == 'bob':
            _, decrypt_err = sess.run([bob_optimizer, decrypt_err_bob], feed_dict = feed_dict)
            bob_decrypt_error = min(bob_decrypt_error, decrypt_err)
        elif network == 'eve':
            _, decrypt_err = sess.run([eve_optimizer, decrypt_err_eve], feed_dict = feed_dict)
            eve_decrypt_error = min(eve_decrypt_error, decrypt_err)

    return alice_decrypt_error, bob_decrypt_error, eve_decrypt_error
text = input('Enter some text: ')
bintext = ' '.join('{0:08b}'.format(ord(x), 'b') for x in text)
print (bintext)
b1  = bintext.replace(" ", "")
#b1 = b1.zfill(48)
pad = len(b1)%16

v1 = np.array([])
for i in range(0, len(b1)):
    v1 = np.append(v1, int(b1[i]))

#apply the padding
for i in range(0, pad):
    v1 = np.append(v1, int(0))
total_len = len(b1) + pad

plaintext_to_Alice = v1.reshape(int(total_len/16), 16)
print('plaintext_to_Alice = ', plaintext_to_Alice)

""" def send_text(message):
    filename = 'text.bin'   
    msg2 = pickle.dumps(message)   
    with open(filename, 'wb') as file_object:
        file_object.write(msg2)
def get_text():
    filename = 'text.bin' 
    with open(filename, 'rb') as file_object:
        bob_output=file_object.read()  
    with open(filename, 'wb'): pass 
    new_bob=str(bob_output)[2: -1]   
    print("bob output",new_bob) 
    return new_bob  """


test_file_msg = "testmsg.txt"
test_file_keys = "testkey.txt"

def test(network, sess):
        alice_decrypt_error, bob_decrypt_error, eve_decrypt_error = 1., 1., 1.
        alice_encrypt_time = 0
        bob_decrypt_time = 0
        bob_output_1 = 0
        
        #bs = 3 #batch_size
        bs = int(total_len/16) #batch_size
        messages, keys = gen_data(n=bs, msg_len=crypto_msg_len, key_len=crypto_key_len)
        #test message to get the code complete
        messages = np.array([
                    [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1], 
                    [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
                   ])


        #feed_dict={msg: messages, key: keys}
        feed_dict={msg: plaintext_to_Alice, key: keys}

        print( 'messages = \n', plaintext_to_Alice)
        print( 'keys = \n', keys)
        
      
        if network == 'alice':
            print('plaintext_to_Alice = ', plaintext_to_Alice)
            start_time = time.time()
            _, decrypt_err, alice_output_1  = sess.run([alice_optimizer, decrypt_err_alice, alice_output], feed_dict = feed_dict)
            end_time = time.time()
            alice_encrypt_time = end_time-start_time    
            alice_decrypt_error = min(alice_decrypt_error, decrypt_err)
           
        elif network == 'bob':
            
            messages_sock =plaintext_to_Alice
            
            feed_dict1 = {msg: messages_sock, key: keys}
            start_time = time.time()
            _, decrypt_err, bob_input_1, bob_output_1 = sess.run([bob_optimizer, decrypt_err_bob, bob_input, bob_output], feed_dict = feed_dict1)
            end_time = time.time()
            bob_decrypt_time = end_time-start_time
            bob_input_1 = np.round(bob_input_1, 2)
            bob_output_1 = np.round(bob_output_1, 2)
            
            print('test bob_input (***Cipher Text***) = \n', bob_input_1, 2)       
            print('test bob_output (***Plain Text***)= \n',bob_output_1, 2)       
            bob_decrypt_error = min(bob_decrypt_error, decrypt_err)
        elif network == 'eve':
            _, decrypt_err = sess.run([eve_optimizer, decrypt_err_eve], feed_dict = feed_dict)
            eve_decrypt_error = min(eve_decrypt_error, decrypt_err)

        return alice_decrypt_error, alice_encrypt_time, bob_decrypt_error, bob_decrypt_time, bob_output_1, eve_decrypt_error

import time
epochs = 10
#sess = tf.InteractiveSession()
sess = tf.compat.v1.Session()

#with tf.Session() as sess:
print('Starting Training Process... ')
start_time = time.time()
train(sess)
end_time = time.time()
print('Time taken for Training (seconds): ', end_time-start_time)


print('alice_errors_train = ', alice_errors)
print('bob_errors_train = ', bob_errors)
print('eve_errors_train = ', eve_errors)
import binascii

#Test the Neural Crypto model
start_time = time.time()
print( 'Testing Alice' )
alice_loss_test, alice_encrypt_time, _, _, _,_ = test('alice', sess)
print('alice_errors_test = ', alice_loss_test)
print( 'Testing Bob' )
_,_, bob_loss_test, bob_decrypt_time, bob_output_1,_ = test('bob', sess)
print('bob_errors_test = ', bob_loss_test)    
print( 'Testing Eve' )
_, _,_,_,_,eve_loss_test = test('eve', sess)
print('eve_errors_test = ', eve_loss_test)
end_time = time.time()
print('Time taken for Testing (seconds): ', end_time-start_time)
#convert to ASCII
b1 = np.around(bob_output_1, decimals=1)
b1 = b1.ravel()
b1 = np.abs(b1)
print('bob_ouput_1 == ', b1)
print('\n')
b2  = np.array2string(b1)
b2 = b2.strip('[')
b2 = b2.strip(']')
b2 = b2.replace(" ", "")
b2 = b2.replace(".", "")
b2 = b2.replace("\n", "")

b2 = '0b'+b2
print('b2 = ', b2)
n = int(b2,2)
str2 = n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(errors='ignore')


print('Bob recovered Plain Text = ', str2)