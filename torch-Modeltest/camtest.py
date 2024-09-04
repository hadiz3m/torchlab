# import the opencv library 
import cv2 
# Import necessary libraries
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy


class mymodel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		
		self.inputimage = nn.Linear(921600,100)
		self.func = nn.ReLU()
		self.prob = nn.Softmax(5)

	def forward(self,x):
		x= self.inputimage(x)
		x= self.func(x)
		x= self.prob(x)
		return x
	

model = mymodel()
model.train()

loss = nn.MSELoss()
optimazer = torch.optim.SGD(model.parameters,lr=0.001)


# define a video capture object 
vid = cv2.VideoCapture(0) 

epoch = 0 

while(True): 
	
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 

	# Display the resulting frame 
	cv2.imshow('frame', frame) 
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	myiimmag = image.reshape(-1)
	# print(myiimmag)
	myiimmag = torch.tensor(myiimmag,dtype=torch.int32)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

	Y_predict = model.forward(myiimmag)

	print(Y_predict)
	l = loss(Y_predict,Y_predict)
	l.backward()

	epoch+=1

	optimazer.step()

	optimazer.zero_grad()

	if epoch % 5 == 0:
		[w,b] = model.parameters()
		print(f' epoch {epoch+1} : w={w[0][0].item():3f} , loss = {l:.8f} ')

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
