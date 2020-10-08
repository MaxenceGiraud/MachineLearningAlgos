import numpy as np
import tensorflow as tf 
from tensorflow.contrib.layers import fully_connected as fc ## Compat Only with tf1 


class VariationalAutoEncododer(AutoEncododer):
    def __init__(self,architecture, d_inputs, learning_rate=1e-4, batch_size=64, d_z=16):
        #architecture is a list of integers describing the number of hidden units per hidden layer
        super().__init__(architecture, d_inputs, learning_rate, batch_size, d_z)
        
    
    def build_decoder(self, z, n_layers, architecture, reuse=False):
        with tf.variable_scope('decoder') as scope:
            if reuse is True:
                scope.reuse_variables()
            #Q3
            layers = []
            layers.append(fc(z,architecture[-1],activation_fn=tf.nn.relu))

            for i in range(1,n_layers):
                layers.append(fc(layers[-1],architecture[n_layers-i-1],activation_fn=tf.nn.relu))

            # esp_x_given_z
            x_hat = fc(layers[-1],d_inputs,activation_fn=None)
            #End of Q3

        return x_hat
    
    def build_network(self, architecture, d_inputs):
        self.x = tf.placeholder(tf.float32,shape=[None, d_inputs])
        n_layers = len(architecture)
        layers = []
        
        #encoder / Q1
        layers.append(fc(self.x,architecture[0],activation_fn=tf.nn.relu))

        for i in range(1,n_layers):
            layers.append(fc(layers[-1],architecture[i],activation_fn=tf.nn.relu))

        esp_z_given_x = fc(layers[-1],self.d_z,activation_fn=None) 
        log_sigma_sq_z_given_x = fc(layers[-1],self.d_z,activation_fn=None) 

        #Sampling z given x / Q2
        epsi = tf.random_normal([self.d_z])
        self.z = tf.math.add(esp_z_given_x, tf.math.multiply((tf.math.exp(0.5*log_sigma_sq_z_given_x)), epsi))

        #decoder
        self.x_hat = self.build_decoder(self.z, n_layers, architecture)        
        
        #Q4
        cross_ent_term = 0.5*tf.math.reduce_sum(tf.square(self.x_hat-self.x))#end of Q4
        
        #Q5
        kl_term = -0.5*tf.reduce_sum(1+0.5*log_sigma_sq_z_given_x - tf.math.exp(log_sigma_sq_z_given_x) - tf.math.square(esp_z_given_x)) #end of Q5
        self.J = cross_ent_term + kl_term
        

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.J)
        self.losses = {
            'cross entropy loss term': cross_ent_term,
            'KL divergence loss term': kl_term
        }
        #for generation / Q8
        self.z_gen = tf.placeholder(tf.float32,shape=[None, self.d_z]) #end of Q8
        self.x_gen = self.build_decoder(self.z_gen, n_layers, architecture, reuse = True)    
        
    # z -> xx
    def generator(self, z):
        x_hat = self.sess.run(self.x_gen, feed_dict={self.z_gen: z})
        return x_hat


class NeuralNetwork():
    
    def __init__(self,architecture, d_inputs, learning_rate=1e-4, batch_size=64, d_z=16):
        #architecture is a list of integers describing the number of hidden units per hidden layer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.d_inputs = d_inputs

        tf.reset_default_graph()
        self.build_network(architecture, d_inputs)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer()) 
        
    def train(self,inputs,n_epochs,batch_size):
        n_inputs = inputs.shape[0]
        for epoch in range(n_epochs):
            n_batches = n_inputs // batch_size
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                x_batch = inputs[iteration*batch_size:(iteration+1)*batch_size]
                self.sess.run(self.train_op, feed_dict={self.x: x_batch})
            loss_train = self.J.eval(feed_dict={self.x: x_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)   
            
class AutoEncododer(NeuralNetwork):
    
    def __init__(self,architecture, d_inputs, learning_rate=1e-4, batch_size=64, d_z=16):
        self.d_z = d_z        
        super().__init__(architecture, d_inputs, learning_rate, batch_size)

    def build_network(self, architecture, d_inputs):
        self.x = tf.placeholder(tf.float32,shape=[None, d_inputs])
        n_layers = len(architecture)
        layers = []
        #encoder
        for i in range(n_layers):
            if (i==0):
                layers.append(fc(self.x,architecture[i],activation_fn=tf.nn.relu))
            else:
                layers.append(fc(layers[i-1],architecture[i],activation_fn=tf.nn.relu))
        self.z = fc(layers[-1],self.d_z,activation_fn=tf.nn.relu)
        #decoder
        for i in range(n_layers):
            if (i==0):
                layers.append(fc(self.z,architecture[n_layers-i-1],activation_fn=tf.nn.relu))
            else:
                layers.append(fc(layers[i-1],architecture[n_layers-i-1],activation_fn=tf.nn.relu))
        self.x_hat = fc(layers[-1],d_inputs,activation_fn=tf.nn.relu)      
        #J
        self.J = tf.reduce_mean(tf.square(self.x_hat - self.x)) 
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.J)
        self.losses = {
            'reconstruction loss': self.J
        }
     
    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat
    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z        
    