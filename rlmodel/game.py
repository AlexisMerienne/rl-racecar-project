from . import preprocessing
from . import double_dqn

import gym
import random
import numpy as np
import matplotlib.pyplot as plt


COEFF_VITESSE = 0.5
NBR_GAME = 10
TIME_RACE = 1000
NBR_ACTION = 4
EPSILON_RATE = 5000


BASH = []

class game:

	def __init__(self):
		self.data = []

	def start(self):

		env = gym.make("CarRacing-v0")
		model = double_dqn.DoubleDQN()
		eps = 0.9
		total_reward_list = [[],[]]
		
		for _race in range(NBR_GAME):

			print("race : ",_race)
		
			total_reward = 0
			state = env.reset()
			img = np.array(state)
			
			p = preprocessing.Prepocessing(np.asarray(state))
			p.image_preprocess()
		
			shape = p.image_pp.shape

			#first game, we define the model
			if _race==0:
				model._model(shape,NBR_ACTION)
				model.get_model_summary()
			else :
				env.close()
				model.train(BASH)
				bash_reset()

			actions = env.action_space.sample()
	
			for _frame in range(TIME_RACE):

				env.render()

				if _frame==0: observation = state


				#prepocessing current state. 
				p_state = preprocessing.Prepocessing(np.asarray(observation))
				p_state.image_preprocess()
				curr_state = p_state.image_pp	

				eGreedy = e_greedy(eps)
				if _race==0 or eGreedy:
					actions = self.random_action()
					#actions = env.action_space.sample()
				else :
					output = model._predict(curr_state)
					s_value,advantage = output[0][0],np.array(output[1][0])
					q_value = model.q_value(s_value,advantage)
					actions = convert_output_to_actions(q_value)
				eps -= (eps/EPSILON_RATE)



				observation, reward, terminated, info = env.step(actions)
				



				img = np.array(observation)
				p_next_state = preprocessing.Prepocessing(img)
				p_next_state.image_preprocess()
				transition_state = p_next_state.image_pp
				
				actions = convert_actions_to_output(actions)
				print(actions)
				

				add_to_BASH(curr_state,actions,reward,transition_state)
				total_reward += reward
			
			total_reward_list[0].append(total_reward)
			total_reward_list[1].append(_race)

		env.close()


		plt.plot(total_reward_list[1],total_reward_list[0])
		plt.show()

	'''
	4 actions : 
		break
		speed + left
		speed + right
		speed + forward
	'''
	def random_action(self):

		direction = random.randint(0,2)
		speed = random.randint(0,1)
		actions = [0,0,0]
		if direction==0:
			actions[0]=-1
		if direction==1 :
			actions[0]=1
		else :
			actions[0]=0
		if speed==0:
			actions[1]=1
		else :
			actions[2]=1

		return actions
		

def add_to_BASH(curr_state,action,reward,transition_state):
	BASH.append({
		"curr_state" : curr_state,
		"action" : action,
		"reward" : reward,
		"transition_state" : transition_state
	})

def bash_reset():
	BASH = []

def convert_actions_to_output(actions):
	output = np.zeros(NBR_ACTION)
	if actions[0] == -1  and actions[1]==1 :
		output[0]=1
	if actions[0] == 1 and actions[1]==1 :
		output[1] = 1
	if actions[0] == 0 and  actions[1]==1:
		output[2] = 1
	if actions[2] == 1:
		output[3] = 1
	
	return output


def convert_output_to_actions(output):

	actions = [0,0,0]
	a  = np.argmax(output)
	if a==0:
		actions=[-1,1,0]
	if a==1:
		actions=[1,1,0]
	if a==2:
		actions=[0,1,0]
	if a==3:
		actions=[0,0,1]

	return actions


def e_greedy(epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return True
    else:
        return False