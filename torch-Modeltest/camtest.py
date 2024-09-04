# import the opencv library 
import cv2 
# Import necessary libraries
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy
import random
import imutils


class mymodel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		
		self.inputimage = nn.Linear(307200,500)
		self.func = nn.ReLU()
		self.inputimage2 = nn.Linear(500,2000)
		self.prob2 = nn.ReLU()
		# self.inputimage3 = nn.Softmax()
		self.inputimage3 = nn.Linear(2000,500)
		self.prob3 = nn.ReLU()
		self.inputimage4 = nn.Linear(500,1)
		self.prob = nn.ReLU()

	def forward(self,x):
		x= self.inputimage(x)
		x= self.func(x)
		x= self.inputimage2(x)
		x= self.prob2(x)
		x= self.inputimage3(x)
		x= self.prob3(x)
		x= self.inputimage4(x)
		x= self.prob(x)
		return x
	

model = mymodel()
model.train()
imwgewith = 640
# CrossEntropyLoss
loss = nn.MSELoss()
# loss = nn.CrossEntropyLoss()
optimazer = torch.optim.Adam(model.parameters(),lr=1e-8)


# define a video capture object 
vid = cv2.VideoCapture(0) 

epoch = 0 

state = torch.tensor([3.0],dtype=torch.float32)
# state = torch.tensor([1.0,0.0,0.0,0.0,0.0],dtype=torch.float32)
name ='start'

while(True): 
	
	# Capture the video frame 
	# by frame 
	optimazer.zero_grad()
	ret, frame = vid.read() 

	# Display the resulting frame 
	cv2.imshow('frame', frame) 

	frame = imutils.resize(frame, width=imwgewith)
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	image = cv2.Canny(image,100,200)
	# image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	myiimmag = image.reshape(-1)
	# print(myiimmag)
	myiimmag = torch.tensor(myiimmag,dtype=torch.float32)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

	Y_predict = model.forward(myiimmag)


	# name = input('name:')
	
	# print(Y_predict)
	l = loss(Y_predict,state)
	l.backward()

	epoch+=1

	optimazer.step()

	

	if epoch % 1 == 0:
		print(f' epoch {epoch+1} , loss = {l:.8f} , output = {Y_predict.item() :.12f} , state = {state.item():.8f}')
		# print(state)
		# print(Y_predict)

	if epoch % 30 == 0:
		# print(f' epoch {epoch+1} , loss = {l:.8f} ')
		foo = ['1', '2', '3', '4', '1']
		name = random.choice(foo)
		if name == '1':
			state = torch.tensor([1.0],dtype=torch.float32)
			# state = torch.tensor([1.0,0.0,0.0,0.0,0.0],dtype=torch.float32)
		elif name == '2':
			state = torch.tensor([2.0],dtype=torch.float32)
			# state = torch.tensor([0.0,1.0,0.0,0.0,0.0],dtype=torch.float32)
		elif name == '3':
			state = torch.tensor([3.0],dtype=torch.float32)
			# state = torch.tensor([0.0,0.0,1.0,0.0,0.0],dtype=torch.float32)
		elif name == '4':
			state = torch.tensor([4.0],dtype=torch.float32)
			# state = torch.tensor([0.0,0.0,0.0,1.0,0.0],dtype=torch.float32)
		elif name == '5':
			state = torch.tensor([5.0],dtype=torch.float32)
			# state = torch.tensor([0.0,0.0,0.0,0.0,1.0],dtype=torch.float32)
	
		print(name)
		print(state)

	# if epoch < 50:
	# 	state = torch.tensor([1.0,0.0,0.0,0.0,0.0],dtype=torch.float32)
	# 	name ='mj'

	# if epoch < 100 and epoch > 50 :
	# 	state = torch.tensor([0.0,1.0,0.0,0.0,0.0],dtype=torch.float32)
	# 	name = 'ali'

	# if epoch < 150  and epoch > 100 :
	# 	state = torch.tensor([0.0,0.0,1.0,0.0,0.0],dtype=torch.float32)
	# 	name ='mahdi'

	# if epoch < 200  and epoch > 150 :
	# 	state = torch.tensor([0.0,0.0,0.0,1.0,0.0],dtype=torch.float32)
	# 	name  ='zz'

	# if epoch < 500  and epoch > 200 :
	# 	break




model.eval()

while(True): 
	
	ret, frame = vid.read() 

	# Display the resulting frame 
	cv2.imshow('frame', frame) 
	frame = imutils.resize(frame, width=imwgewith)
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	image = cv2.Canny(image,100,200)	
	# image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	myiimmag = image.reshape(-1)
	# print(myiimmag)
	myiimmag = torch.tensor(myiimmag,dtype=torch.float32)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

	Y_predict = model.forward(myiimmag)
	print(Y_predict)

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
