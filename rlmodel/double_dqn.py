import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input,Model
from tensorflow.keras.optimizers  import Adam
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K


class DoubleDQN:

    input = []
    shape = ()

    def __init__(self) :

    
        print("-- ann_network constructor --")
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.005
        self.batch_size = 124

    def mean_tensor(self,vect):
        return tf.reduce_mean(vect)

    def concatenate(self,x):
        advantage_value,advantage_mean,state_value = x[0],x[1],x[2]
        output = advantage_value - advantage_mean + state_value
        return output

    def _model(self,shape,nbr_action):

        shape = (shape[0],shape[1],1)

        input = layers.Input(shape=shape,
                               dtype='float32', name='input')
        conv = layers.Conv2D(3,2,activation='relu',input_shape=shape,name="convolutional_layer")(input)

        pool = layers.MaxPooling2D((2, 2),name="pooling")(conv)

        flatten = layers.Flatten(name="just_flatting")(pool)

        h_state_value = layers.Dense(32, activation="relu",name="hidden_layer_state_value")(flatten)
        h_advantage_function = layers.Dense(32, activation="relu",name="hidden_layer_advantage_function")(flatten)


        output_state_value = layers.Dense(1,activation='sigmoid',name="output_state_value")(h_state_value)
        output_advantage_function = layers.Dense(nbr_action,activation='sigmoid',name="output_advantage_fonction")(h_advantage_function)
      
        '''
        tentative de faire un modèle dont l'ouptut est la q-value
        '''
        #mean_advantage_value = layers.Lambda(self.mean_tensor)(output_advantage_function)

        #output = layers.Lambda(self.concatenate)([output_advantage_function,mean_advantage_value,output_state_value])

                
        self.model = Model(input, [output_state_value,output_advantage_function])
        plot_model(
            self.model,
            to_file="../model.png",
            show_shapes=True,expand_nested=True)

    
        self.model.compile(optimizer=Adam(lr=self.lr), loss='mse')

    def get_model_summary(self):
        self.model.summary()

    '''
    
    '''
    def train(self,BASH):
        print(" --- fitting ----")

        buffer = BASH

        np.random.shuffle(buffer)
        batch_sample = buffer[0:self.batch_size]

        for _step in batch_sample:
            _s = np.expand_dims(_step["curr_state"], axis=0)
            _s = np.expand_dims(_s,axis=3)
            output = self.model.predict(_s)

            s_value,advantage = output[0][0],np.array(output[1][0])
            
            #q_value = s_value + (advantage_value - moyenne(advantage))
            q_value = self.q_value(s_value,advantage)

            #print("Q_CURRENT_STATE = ",q_current_state)
            #print("BASH Action",_step["action"])

            next_state = np.expand_dims(_step["transition_state"], axis=0)
            next_state = np.expand_dims(next_state,axis=3)
            # We compute the Q-target using Bellman optimality equation
            output_target = self.model.predict(next_state)
            s_value_target,advantage_target = output_target[0][0],np.array(output_target[1][0])

            q_target =_step["reward"] + self.gamma*np.max(self.q_value(s_value_target,advantage_target))
            #np.argmax(_step["action"]) don't answer the issue.
            q_value[np.argmax(_step["action"])] = q_target

            s_value_output = float(np.mean(q_value))
            
            # train the model
            self.model.fit(_s,[np.array([s_value_output]),np.array([q_value])],verbose=0)

    def q_value(self,advantage,s_value):
        return advantage + (s_value - np.mean(advantage))
        
    def _predict(self,input):
        input = np.expand_dims(input, axis=0)
        return self.model.predict(input)
        