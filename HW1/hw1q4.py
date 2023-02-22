import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


#4.a
#calculate cylic place
def cycle_shift(i, edge):
 place = i
 if ((i +1) == edge):
  place = 0
 return place

#shift by unatural number [0,1)
def bilinear_displacement(dx, dy, image):
 """
 Calculate the displacement of a pixel using a bilinear interpolation.
 :param dx: the displacement in the x direction. dx in rang [0,1).
 :param dy: the displacement in the y direction. dy in rang [0,1).
 :param image: The image on which we preform the cyclic displacement
 :return:
 displaced_image: The new displaced image
 """

 # ====== YOUR CODE: ======
 displaced_image = image
 for i in range(0, displaced_image.shape[0]):
  place_i = cycle_shift(i, displaced_image.shape[0])
  for j in range(0, displaced_image.shape[1]):
   place_j = cycle_shift(j, displaced_image.shape[1])
   displaced_image[i][j] = (((1-dx)*image[i][j]+dx*image[place_i][j])*(1-dy)+((1-dx)*image[i][place_j]+dx*image[place_i][place_j])*dy)


 # ========================
 return displaced_image



#4.b
#shift picture
def genral_displacement(dy,dx , image):
 """
 Calculate the displacement of a pixel using a bilinear interpolation.
 :param dx: the displacement in the x direction.
 :param dy: the displacement in the y direction.
 :param image: The image on which we preform the cyclic displacement
 :return:
 displaced_image: The new displaced imag
 """
 # ====== YOUR CODE: ======
 #devide to natural and unatural part
 dx_natural = int(dx) % image.shape[0]
 dx_fraction = dx % 1
 dy_natural = int(dy) % image.shape[1]
 dy_fraction = dy % 1

#shift by natura number
 new_img = np.zeros(image.shape)
 displaced_image_natural = new_img.astype(np.uint8)
 for i in range(0, displaced_image_natural.shape[0]):
  for j in range(0, displaced_image_natural.shape[1]):
   displaced_image_natural[i][j] = image[(i - dx_natural) % displaced_image_natural.shape[0]][(j - dy_natural) % displaced_image_natural.shape[1]]




 #shift by the unatural part
 displaced_image = bilinear_displacement(dx_fraction, dy_fraction, displaced_image_natural)

 # ========================
 return displaced_image

#4.c
img_cameraman = cv2.imread('../given_data/cameraman.jpg') #open image
gray_cameraman = cv2.cvtColor(img_cameraman, cv2.COLOR_BGR2GRAY) #convert to gray
plt.imshow(gray_cameraman, cmap='gray')
plt.title('Cameraman image in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


# shift cameraman by a sprcific shift
displaced_cameraman = genral_displacement(150.7, 110.4 , gray_cameraman)
plt.imshow(displaced_cameraman, cmap='gray')
plt.title('Cameraman image after displacement')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#4.d
img_Ryan = cv2.imread('../given_data/Ryan.jpg') #open image
gray_Ryan = cv2.cvtColor(img_Ryan, cv2.COLOR_BGR2GRAY) #convert to gray
plt.imshow(gray_Ryan, cmap='gray')
plt.title('Rayen image in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

# finding important sizes for the mask
width, hieght = gray_Ryan.shape

center_h = 44
center_w = 370
radius = 203 - 44

# creating a rectangle mask
rectangle = np.zeros_like(gray_Ryan)
cv2.rectangle(rectangle, (0, center_h), (width ,hieght), 255, -1)

# creating a circle mask
circle = np.zeros_like(gray_Ryan)
cv2.circle(circle, (center_w, center_h), radius, 255, -1)

# creating half circle mask (using AND operator between rectangle and circle)
mask1 = cv2.bitwise_and(rectangle, circle)

# Mask input image with half circle mask
ryan_win = cv2.bitwise_and(gray_Ryan, mask1)
plt.imshow(ryan_win, cmap='gray')
plt.title('ryan win')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()




#4.e
def rotating_img(image,theta):

 #====== YOUR CODE: ======
 #calculate the center and change indexes acordingly
 outcome = np.zeros(image.shape)
 for i in range(0, image.shape[0]):
  for j in range(0, image.shape[1]):
   shifted_y = round((image.shape[0])/2) -i
   shifted_x = round((image.shape[1])/2) -j
   #calculate new place after rotating
   x = round(shifted_x * np.cos(theta) + shifted_y * np.sin(theta))
   y = round(-1*shifted_x * np.sin(theta) + shifted_y * np.cos(theta))

   #shift back
   y = round(image.shape[0]/2) - y
   x = round(image.shape[1]/2) - x

   # assign new values
   if 0 <= x < image.shape[1] and 0 <= y < image.shape[0] and x >= 0 and y >= 0:
     outcome[y, x] = image[i, j]

 rotated_image = outcome.astype(np.uint8)


 return rotated_image


#rotate by 45 degrees
image1 = rotating_img(ryan_win, (np.pi)/4)
plt.imshow(image1, cmap='gray')
plt.title('angle pi/4')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#rotate by 60 degrees
image2 = rotating_img(ryan_win, (np.pi)/3)
plt.imshow(image2, cmap='gray')
plt.title('angle pi/3')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#rotate by 90 degrees
image3 = rotating_img(ryan_win, (np.pi)/2)
plt.imshow(image3, cmap='gray')
plt.title('angle pi/2')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()






