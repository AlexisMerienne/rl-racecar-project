from . import preprocessing
from . import ann_network

import gym
import random
import numpy as np


COEFF_VITESSE = 0.5
NBR_GAME = 5
TIME_RACE = 100


BASH = {
	"input" : [],
	"output" : []
}

class game:

	def __init__(self):
		self.data = []

	def start(self):

		env = gym.make("CarRacing-v0")
		model = ann_network.Ann_Network()
		
		for _race in range(NBR_GAME):
		
			state = env.reset()

			img = np.array(state)
			p = preprocessing.Prepocessing(state)
			p.image_preprocess()
		
			shape = p.image_pp.shape
			state = p.image_pp

			#first game, we define the model
			if _race==0:
				model._model(shape)
				model.get_model_summary()
			else :
				model.fit_model(np.array(BASH["input"]),np.array(BASH["output"]))
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

				
				observation, reward, terminated, info = env.step(actions)
				
				img = np.array(observation)
				p = preprocessing.Prepocessing(img)
				p.image_preprocess()
				observation = p.image_pp
				p.get_pos_car()
				is_off_track = p.get_pos_car()

				output = convert_actions_to_output(actions)
				

				BASH["input"].append(p.image_pp)
				BASH["output"].append(np.array(output))

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
		



def bash_reset():
	BASH["input"],BASH["ouput"] = [],[]

def convert_actions_to_output(actions):
	output = np.zeros(4)
	if actions[0] == -1:
		output[0]=1
	else : output[1] = 1
	if actions[1]==1:
		output[2]=1
	else : output[3]=1
	
	return output


def convert_output_to_actions(output):

	actions = [0,0,0]

	direction_ar = np.array(output)[:2]
	speed_ar = np.array(output)[2:4]
	direction = np.argmax(direction_ar)
	speed = np.argmax(speed_ar)
	
	if direction==0:actions[0]=-1
	else : actions[0]=1
	if speed==0:actions[1]=1
	else:actions[2]=1

	print("output : ",output," -- actions : ",actions)
	

	return actions