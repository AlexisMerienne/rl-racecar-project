from . import preprocessing
from . import double_dqn

import gym
import random
import numpy as np


COEFF_VITESSE = 0.5
NBR_GAME = 5
TIME_RACE = 100	
NBR_ACTION = 4


BASH = []

class game:

	def __init__(self):
		self.data = []

	def start(self):

		env = gym.make("CarRacing-v0")
		model = double_dqn.DoubleDQN()
		
		for _race in range(NBR_GAME):
		
			state = env.reset()
			img = np.array(state)
			
			p = preprocessing.Prepocessing(np.asarray(state))
			p.image_preprocess()
		
			shape = p.image_pp.shape
			state = p.image_pp

			#first game, we define the model
			if _race==0:
				model._model(shape,NBR_ACTION)
				model.get_model_summary()
			else :
				model.train(BASH)
				bash_reset()

			actions = env.action_space.sample()
	
			for _frame in range(TIME_RACE):

				env.render()

				if _frame==0: observation = state


				if _race==0:
					actions = self.random_action()
				else : 
					output = model._predict(observation)[0]
					actions = convert_output_to_actions(output)

				curr_state = p.image_pp


				observation, reward, terminated, info = env.step(actions)
				

				img = np.array(observation)
				p = preprocessing.Prepocessing(img)
				p.image_preprocess()
				observation = p.image_pp
				p.get_pos_car()
				is_off_track = p.get_pos_car()

				actions = convert_actions_to_output(actions)
				transition_state = np.array(observation)
				

				add_to_BASH(curr_state,actions,reward,transition_state)
				

				if is_off_track==1:break
									

		env.close()

		img = np.array(observation)
		p = preprocessing.Prepocessing(img)
		p.image_preprocess()
		print(p.image_pp.shape)
		p.plot_img(p.image_pp)

	def random_action(self):

		direction = random.randint(0,1)
		speed = random.randint(0,1)
		actions = [0,0,0]
		if direction==0:
			actions[0]=-1
		else :
			actions[0]=1
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
	output = np.zeros((2,2))
	if actions[0] == -1:
		output[0,0]=1
	else : output[0,1] = 1
	if actions[1]==1:
		output[1,0]=1
	else : output[1,1]=1
	
	return output


def convert_output_to_actions(output):

	actions = [0,0,0]

	direction_ar = np.array(output)[0]
	speed_ar = np.array(output)[1]
	direction = np.argmax(direction_ar)
	speed = np.argmax(speed_ar)
	
	if direction==0:actions[0]=-1
	else : actions[0]=1
	if speed==0:actions[1]=1
	else:actions[2]=1

	print("output : ",output," -- actions : ",actions)
	

	return actions