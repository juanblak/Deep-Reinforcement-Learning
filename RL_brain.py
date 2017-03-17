
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from datetime import datetime

np.random.seed(0)
tf.set_random_seed(0)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            hidden_layers=[10, 10],
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.episode=0
        self.time=time.time()
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.hidden_layers = hidden_layers
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_features*2+2)))

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()

        #if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
       
        # tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.cost_time=[]
        self.cost_time2=[]
        self.sum=[]
        self.sum2=[]

        self.pastop=[1,0,3,2]
        self.arcount=0
        self.ascount=0



    def _build_net(self):
        # create eval and target net weights and biases separately
        self._eval_net_params = []
        self._target_net_params = []

        # build evaluate_net
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):
            self.q_eval = self._build_layers(self.s, self.n_actions, trainable=True)
            with tf.name_scope('loss'):
                self.loss = tf.reduce_sum(tf.square(self.q_target - self.q_eval))
                tf.scalar_summary('loss',self.loss)
            with tf.name_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
                #GradientDescentOptimizer 11min>
                #RMSPropOptimizer 182.8
                #AdadeltaOptimizer 9min>
                #AdagradOptimizer  --
                #AdagraDAOptimizer
                #MomentumOptimizer  --
                #AdamOptimizer  225 no
                #FtrlOptimizer
                #ProximalGradientDescentOptimizer
                #proximalAdagradOptimizer

        # build target_net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            self.q_next = self._build_layers(self.s_, self.n_actions, trainable=False)

    def _build_layers(self, inputs, action_size, trainable):
        layers_output = [inputs]
        for i, n_unit in enumerate(self.hidden_layers):
            with tf.variable_scope('layer%i' % i):
                output = self._add_layer(
                    layers_output[i],
                    in_size=layers_output[i].get_shape()[1].value,
                    out_size=n_unit,
                    activation_function=tf.nn.relu,
                    trainable=trainable,
                )
                layers_output.append(output)
        with tf.variable_scope('output_layer'):
            output = self._add_layer(
                layers_output[-1],
                in_size=layers_output[-1].get_shape()[1].value,
                out_size=action_size,
                activation_function=None,
                trainable=trainable
            )
        return output

    def _add_layer(self, inputs, in_size, out_size, activation_function=None, trainable=True):
        # create weights and biases
        Weights = tf.get_variable(
            name='weights',
            shape=[in_size, out_size],
            trainable=trainable,
            initializer=tf.truncated_normal_initializer(mean=0., stddev=0.3)
        )
        biases = tf.get_variable(
            name='biases',
            shape=[out_size],
            initializer=tf.constant_initializer(0.1),
            trainable=trainable
        )

        # record parameters
        if trainable is True:
            self._eval_net_params.append([Weights, biases])
        else:
            self._target_net_params.append([Weights, biases])

        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # activation function
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        #print(transition.shape) #(6,)

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation,pa):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)

        else:

            while True:
                r=np.random.uniform()
                if r < 0.5 :
                    action=3
                else:
                    action=1
                self.ascount=self.ascount+1


                if pa==-1:
                    break

                elif self.pastop[pa]!=action:
                    break

                elif self.pastop[pa]==action:
                    self.arcount=self.arcount+1
                    continue

        return action

    def choose_action2(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        return action

    def _replace_target_params(self):
        replace_ops = []
        for layer, params in enumerate(self._eval_net_params):
            replace_op = [tf.assign(self._target_net_params[layer][W_b], params[W_b]) for W_b in range(2)]
            replace_ops.append(replace_op)
        self.sess.run(replace_ops)

    def learn(self,episode):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        batch_memory = self.memory.sample(self.batch_size) \
            if self.memory_counter > self.memory_size \
            else self.memory.iloc[:self.memory_counter].sample(self.batch_size, replace=True)
        #print(batch_memory.shape) #(32,6)
 
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory.iloc[:, -self.n_features:],
                self.s: batch_memory.iloc[:, :self.n_features]
            })
        #print(q_eval.shape)#(32,4)
        #print(self.n_features)#2

        #print(batch_memory,q_eval,q_next)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        q_target[np.arange(self.batch_size, dtype=np.int32), batch_memory.iloc[:, self.n_features].astype(int)] = \
            batch_memory.iloc[:, self.n_features+1] + self.gamma * np.max(q_next, axis=1)


        #print(np.max(q_next, axis=1))

        

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory.iloc[:, :self.n_features],
                                                self.q_target: q_target})




        if self.episode!=episode:
            self.cost_his.append(self.cost)
            t1=time.time()
            self.cost_time.append(t1-self.time)
            self.cost_time2.append(t1)

            self.time=t1
            self.episode=episode
            self.sum.append(np.sum(q_target))
            # self.sum2.append(np.sum(q_target)-np.sum(q_eval))



        # increasing epsilon
        #self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure(1) # 创建图表1
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.figure(2) 
        plt.plot(np.arange(len(self.sum)), self.sum,'o')
        plt.figure(3) 
        plt.plot(np.arange(len(self.cost_time)), self.cost_time)
        plt.figure(4) 
        plt.plot(np.arange(len(self.cost_time2)), self.cost_time2)
        # plt.plot(np.arange(len(self.sum2)), self.sum2)
        plt.show()

    def write_cost(self):
        import xlwt
        file=xlwt.Workbook()
        table=file.add_sheet('1')
        for i in np.arange(len(self.cost_his)):
            table.write(i,0,self.cost_his[i])
        filepath='home/jane/Desktop/MY_RL/10(1-path)9/1.xls'

        file.save(filepath)
