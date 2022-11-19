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
        direction = random.randint(0,self.n)
        speed = random.randint(0,self.n-1)
        actions[0] = ((2 * direction) / self.n ) - 1
        if speed==0:actions[2]=self.break_value
        else :actions[1]=speed

        return actions



    '''
    mapping actions of shape [3] to output of shape (3,n)
    '''
    def mapping_to_input(self,actions):
        input = np.zeros((2,self.n))
        if actions[0]<=0:
            input[0,round(self.n/2*(actions[0]+1))] = 1
        elif actions[0]>0:
            input[0,round(self.n/2*(actions[0]+1)-1)] = 1

        
        if actions[1]!=0:
            input[1,actions[1]] = 1
        if actions[2]!=0:
            input[1,0] = 1

        one_hot_input = np.zeros(((self.n) * (self.n)))
        for i in range(self.n):
            for j in range(self.n):
                if input[0,i]==input[1,j] == 1:
                    one_hot_input[j*(self.n)+i]=1
        return one_hot_input

    def mapping_to_actions(self,output):
        actions = [0,0,0]
        out = np.zeros((2,self.n))
        action_one_hot = np.argmax(np.array(output))
        j = action_one_hot // self.n
        i = action_one_hot % self.n
        out[0,i]=1
        out[1,j]=1

        actions[0] = np.argmax(out[0])*2/self.n - 1
       
        if out[1,0]!=0:actions[2]=self.break_value
        else:
            actions[1] = j
        return actions



