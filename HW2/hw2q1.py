import numpy as np
import matplotlib.pyplot as plt
import cv2


#1.a
img = cv2.imread('../given_data/puppy.jpg') #open image
puppy_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray-scale
plt.imshow(puppy_gray, cmap='gray')
plt.title('image of puppy in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


#1.b
#show histogram of puppy
plt.hist(puppy_gray.ravel(),256,[0,256])
plt.title('hiastogram of the puppy')
plt.xlabel("values")
plt.ylabel("amount in picture")
plt.show()




#1.c
def gamma_correction(img, gamma):
 """
 Perform gamma correction on a grayscale image.
 :param img: An input grayscale image - ndarray of uint8 type.
 :param gamma: the gamma parameter for the correction.
 :return:
 gamma_img: An output grayscale image after gamma correction -
 uint8 ndarray of size [H x W x 1].
 """
 # ====== YOUR CODE: ======
 new_img = np.zeros(img.shape)
 for i in range(0, img.shape[0]):
  for j in range(0, img.shape[1]):
      new_img[i][j] = ((img[i][j]/255) ** gamma)*255
 gamma_img = new_img.astype(np.uint8)
 # ========================
 return gamma_img

puppy1 = gamma_correction(puppy_gray, 0.5)
plt.imshow(puppy1, cmap='gray')
plt.title('image of puppy with gamma = 0.5')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

plt.hist(puppy1.ravel(),256,[0,256])
plt.title('hiastogram of the puppy with gamma = 0.5')
plt.xlabel("values")
plt.ylabel("amount in picture")
plt.show()

puppy2 = gamma_correction(puppy_gray, 1.5)
plt.imshow(puppy2, cmap='gray')
plt.title('image of puppy with gamma = 1.5')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

plt.hist(puppy2.ravel(),256,[0,256])
plt.title('hiastogram of the puppy with gamma = 1.5')
plt.xlabel("values")
plt.ylabel("amount in picture")
plt.show()

