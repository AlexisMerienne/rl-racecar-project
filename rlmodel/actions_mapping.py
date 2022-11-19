import numpy as np
import random


'''
Cette classe permet de créer un mapping entre l'espace d'action de l'environnement,
qui est un espace d'action continue, et l'output du modele dual DQN, qui est un vecteur de taille finie


'''
class Mapping_Action():

    n = 1
    break_value = 1

    def __init__(self,n,break_value) :
        self.n = n
        self.break_value=break_value

    def generate_random_action(self):
        actions = [0,0,0]
        direction = random.randint(0,2*self.n)
        speed = random.randint(0,self.n)

        actions[0] = (direction / self.n) -1
        if speed==0:actions[2]=self.break_value
        else :actions[1]=speed

        return actions



    '''
    mapping actions of shape [3] to output of shape (3,n)
    '''
    def mapping_to_input(self,actions):
        input = np.zeros((3,self.n+1))

        if actions[0]<0:
            input[0,int(-actions[0]*self.n)] = 1
        if actions[0]>0:
            input[1,int(actions[0]*self.n)]=1
        if actions[1]!=0:
            input[2,actions[1]] = 1
        if actions[2]!=0:
            input[2,0] = 1

        input = input.reshape(3*(self.n+1))
        return input

    def mapping_to_actions(self,output):
        actions = [0,0,0]
        out = output.reshape((3,self.n+1))
        if np.sum(out[0])>np.sum(out[1]):
            actions[0] = - np.argmax(out[0])/self.n
        else : 
            actions[0] = np.argmax(out[1])/self.n
        if out[2,0]!=0:actions[2]=self.break_value
        else: actions[1] = np.argmax(out[2]-1)

        return actions
    

