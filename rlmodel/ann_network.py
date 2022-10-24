from pyexpat import model
from typing import Concatenate
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Concatenate
from tensorflow.keras import utils



class Ann_Network:

    input = []
    shape = ()
    model

    def __init__(self) :
        print("-- ann_network constructor --")

    def _model(self,shape):

        shape = (shape[0],shape[1],1)

        inputs = keras.Input(shape=shape)

        conv = Conv2D(2,3,activation='relu',input_shape=shape)(inputs)
        x = MaxPooling2D((2, 2))(conv)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)

        outputs_direction = Dense(2,activation='softmax')(x)
        outputs_speed = Dense(2,activation='softmax')(x)

        outputs = Concatenate(name="concatenate")([outputs_direction,outputs_speed])

        
        self.model = keras.Model(inputs, outputs)
    
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    def get_model_summary(self):
        self.model.summary()


    def fit_model(self,X_train,y_train):
        print(X_train.shape," - ",y_train.shape)
        self.model.fit(X_train,y_train, epochs=30, validation_split=0.33)
        
    def _predict(self,input):
        input = np.expand_dims(input, axis=0)
        return self.model.predict(input)
        