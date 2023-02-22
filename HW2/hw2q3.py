import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#3.a
keyboard = cv2.imread('../given_data/keyboard.jpg') #open image
gray_keyboard= cv2.cvtColor(keyboard, cv2.COLOR_BGR2GRAY) #convert to grey
plt.imshow(gray_keyboard, cmap='gray')
plt.title('keyboard image in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

kernel_ver = np.ones((8,1),np.uint8)

kernel_hor = np.ones((1,8),np.uint8)

erosion_ver = cv2.erode(gray_keyboard,kernel_ver,iterations = 1)
plt.imshow(erosion_ver, cmap='gray')
plt.title('verticl kernel erosion in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

erosion_hor = cv2.erode(gray_keyboard,kernel_hor,iterations = 1)
plt.imshow(erosion_hor, cmap='gray')
plt.title('horozontal kernel erosion in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

sum_keyboard= cv2.add(erosion_hor,erosion_ver)
plt.imshow(sum_keyboard, cmap='gray')
plt.title('sum of 2 erosion in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


sum_keyboard_binary = np.zeros(sum_keyboard.shape,np.uint8)
for i in range(0, sum_keyboard.shape[0]):
    for j in range(0, sum_keyboard.shape[1]):
        if (sum_keyboard[i][j] <= 0.2 * 255):
            sum_keyboard_binary[i][j] = 0
        else:
            sum_keyboard_binary[i][j] = 255

plt.imshow(sum_keyboard_binary, cmap='gray')
plt.title('sum of 2 erosion with 2 values only in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#3.b
not_sum_keyboard = cv2.bitwise_not(sum_keyboard_binary)


median_blur = cv2.medianBlur(not_sum_keyboard,9)
plt.imshow(median_blur, cmap='gray')
plt.title('median blurr in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#3.c
kernel_8 = np.ones((8,8),np.uint8)
erosion_8 = cv2.erode(median_blur,kernel_8,iterations = 1)
plt.imshow(erosion_8, cmap='gray')
plt.title('erosion on median blurr in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#3.d
sum_keyboard_binary_uint = np.zeros(sum_keyboard_binary.shape,np.uint8)
for i in range(0, sum_keyboard_binary.shape[0]):
    for j in range(0, sum_keyboard_binary.shape[1]):
        if (sum_keyboard_binary[i][j] == 255):
            sum_keyboard_binary_uint[i][j] = 1


intersection =cv2.bitwise_and(gray_keyboard,erosion_8)
plt.imshow(intersection, cmap='gray')
plt.title('intersection median blurr in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

k = np.array([[0, -1, 0],
            [-1, 5 ,-1],
             [0, -1 ,0]])

k.astype(np.uint)
sharpen =cv2.filter2D(intersection,-1,k)
plt.imshow(sharpen, cmap='gray')
plt.title('sharpen')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

final_keyboard = np.zeros(sum_keyboard.shape,np.uint8)
for i in range(0, sum_keyboard.shape[0]):
    for j in range(0, sum_keyboard.shape[1]):
        if (sharpen[i][j] <= 0.6 * 255):
            final_keyboard[i][j] = 0
        else:
            final_keyboard[i][j] = 255

plt.imshow(final_keyboard, cmap='gray')
plt.title('final keyboard')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()
