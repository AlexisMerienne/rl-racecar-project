import numpy as np


import tensorflow.keras.backend as K
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Activation,Reshape
from tensorflow.keras.optimizers  import Adam


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


    def _model(self,shape,nbr_action):

        shape = (shape[0],shape[1],1)

        inputs = Input(shape=shape)

        conv = Conv2D(2,3,activation='relu',input_shape=shape)(inputs)

        x = MaxPooling2D((2, 2))(conv)
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation('relu')(x)
        x = Dense(16)(x)
        x = Activation('relu')(x)
        x = Dense(16)(x)
        x = Activation('relu')(x)

        outputs_Dense = Dense(nbr_action,activation='linear')(x)
        outputs = Reshape((nbr_action//2, nbr_action//2), input_shape=(nbr_action,))(outputs_Dense)

        
        self.model = Model(inputs, outputs)
    
        self.model.compile(optimizer=Adam(lr=self.lr), loss='mse')

    def get_model_summary(self):
        self.model.summary()

    '''
    
    '''
    def train(self,BASH):

        buffer = BASH

        np.random.shuffle(buffer)
        batch_sample = buffer[0:self.batch_size]

        for _step in batch_sample:
            _s = np.expand_dims(_step["curr_state"], axis=0)
            _s = np.expand_dims(_s,axis=3)
            q_current_state = self.model.predict(_s)

            print("Q_CURRENT_STATE = ",q_current_state)
            print("BASH Action",_step["action"])

            next_state = np.expand_dims(_step["transition_state"], axis=0)
            next_state = np.expand_dims(next_state,axis=3)
            # We compute the Q-target using Bellman optimality equation
            q_target=[0,0]
            q_target[0] =_step["reward"] + self.gamma*np.max(self.model.predict(next_state)[0,0])
            q_target[1] =_step["reward"] + self.gamma*np.max(self.model.predict(next_state)[0,1])

            #np.argmax(_step["action"]) don't answer the issue. 
            q_current_state[0,0][np.argmax(_step["action"][0])] = q_target[0]
            q_current_state[0,1][np.argmax(_step["action"][1])] = q_target[1]
            
            # train the model
            self.model.fit(_s, q_current_state, verbose=0)


        
    def _predict(self,input):
        input = np.expand_dims(input, axis=0)
        return self.model.predict(input)
        