import numpy as np

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