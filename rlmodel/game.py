from . import preprocessing
from . import double_dqn
from . import actions_mapping

import gym
import random
import numpy as np
import matplotlib.pyplot as plt


COEFF_VITESSE = 0.5
NBR_GAME = 200
TIME_RACE = 5000
EPSILON_RATE = 100000
BREAK_RATE = 5
N_ACTION = 10


BASH = []

class game:

	def __init__(self):
		self.data = []

	def start(self):

		env = gym.make("CarRacing-v0")
		model = double_dqn.DoubleDQN()
		eps_init = 0.9
		total_reward_list = [[],[]]
		
		for _race in range(NBR_GAME):

			print("race : ",_race)
		
			total_reward = 0
			state = env.reset()
			img = np.array(state)
			
			p = preprocessing.Prepocessing(np.asarray(state))
			p.image_preprocess()
		
			shape = p.image_pp.shape
			eps = eps_init * (NBR_GAME-_race)/NBR_GAME
			print(eps)

			#first game, we define the model
			if _race==0:
				model._model(shape,N_ACTION*N_ACTION)
				model.get_model_summary()
			else :
				env.close()
				model.train(BASH)
				bash_reset()

			actions = env.action_space.sample()
			out_of_track = 0
	
			for _frame in range(TIME_RACE):
			
	
				env.render()
				

				if _frame==0: observation = state


				#prepocessing current state. 
				p_state = preprocessing.Prepocessing(np.asarray(observation))
				p_state.image_preprocess()
				curr_state = p_state.image_pp	

				eGreedy = e_greedy(eps)
				mapping_actions = actions_mapping.Mapping_Action(N_ACTION,BREAK_RATE)

				if _race==0 or eGreedy:
					actions = mapping_actions.generate_random_action()
					#actions = env.action_space.sample()
				else :
					output = model._predict(curr_state)
					s_value,advantage = output[0][0],np.array(output[1][0])
					q_value = model.q_value(s_value,advantage)
					actions = mapping_actions.mapping_to_actions(q_value)
					
			



				observation, reward, terminated, info = env.step(actions)
				



				img = np.array(observation)
				p_next_state = preprocessing.Prepocessing(img)
				p_next_state.image_preprocess()
				if p_next_state.get_pos_car()==1:
					out_of_track+= 1
					reward=-1
				else:
					out_of_track=0


				
				transition_state = p_next_state.image_pp
				
				actions = mapping_actions.mapping_to_input(actions)
				#si la voiture est en dehors du circuit, cela ne sert à rien d'enregistrer les frames. 
				if out_of_track<20:
					add_to_BASH(curr_state,actions,reward,transition_state)
				total_reward += reward
				
				eps -= (eps/EPSILON_RATE)
				_frame+=1
				
				#Si la voiture est en dehors de la route depuis trop longtemps, on arrête la course. 
				if out_of_track>1000:
					break
				
			total_reward_list[0].append(total_reward)
			total_reward_list[1].append(_race)

		env.close()

		model.save_model()
		plt.plot(total_reward_list[1],total_reward_list[0])
		plt.xlabel("episode")
		plt.ylabel("reward")
		plt.show()

	

		

def add_to_BASH(curr_state,action,reward,transition_state):
	BASH.append({
		"curr_state" : curr_state,
		"action" : action,
		"reward" : reward,
		"transition_state" : transition_state
	})

def bash_reset():
	BASH = []





def e_greedy(epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return True
    else:
        return False