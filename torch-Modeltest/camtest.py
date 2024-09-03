# # program to capture single image from webcam in python 

# # importing OpenCV library 
# import cv2 as cv

# # initialize the camera 
# # If you have multiple camera connected with 
# # current device, assign a value in cam_port 
# # variable according to that 
# cam_port = 0
# cam = cv.VideoCapture(cam_port) 

# # reading the input using the camera 
# result, image = cam.read() 

# # If image will detected without any error, 
# # show result 
# if result: 

# 	# showing result, it take frame name and image 
# 	# output 
# 	cv.imshow("GeeksForGeeks", image) 

# 	# saving image in local storage 
# 	cv.imwrite("GeeksForGeeks.png", image) 

# 	# If keyboard interrupt occurs, destroy image 
# 	# window 
# 	cv.waitKey(0) 
# 	cv.destroyWindow("GeeksForGeeks") 

# # If captured image is corrupted, moving to else part 
# else: 
# 	print("No image detected. Please! try again") 


# import the opencv library 
import cv2 
# Import necessary libraries
import torch
from PIL import Image
import torchvision.transforms as transforms

# Read a PIL image
# image = Image.open('iceland.jpg')



# define a video capture object 
vid = cv2.VideoCapture(0) 

while(True): 
	
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 

	# Display the resulting frame 
	# cv2.imshow('frame', frame) 


	# Define a transform to convert PIL 
	# image to a Torch tensor
	transform = transforms.Compose([
		transforms.ToTensor()
		# transforms.PILToTensor()
	])

	# transform = transforms.PILToTensor()
	# Convert the PIL image to Torch tensor

	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	imagePIL = Image.fromarray(image)
	# imagePIL.show()

	img_tensor = transform(image)

	# print the converted Torch tensor
	print(img_tensor)

	# the 'q' button is set as the 
	# quitting button you may use any 
	# desired button of your choice 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
