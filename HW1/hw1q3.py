import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# #3.a
#create new graph
x = np.arange(-256,256, 1)
X, Y = np.meshgrid(x, x)

# create the function
f_x_y = np.sin(X*2*np.pi*(5/512))+np.sin(Y*2*np.pi*(40/512))+np.sin(2*np.pi*(X+Y)*(2/512))

plt.imshow(f_x_y, cmap='gray')
plt.title('F1(x,y) after sampling with dx=1 dy=1')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#3.b
#convert function to fft
img_fft = np.fft.fft2(f_x_y)
img_fft_shift = np.fft.fftshift(img_fft)
plt.imshow(np.log(1 + np.abs(img_fft_shift)), cmap='gray')
plt.title('F1(X,Y) in the frequency space - FFT')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#3.c

# create a new graph with 10 times less pixels
x = np.arange(-256,256, 10)
X, Y = np.meshgrid(x, x)
f_x_y = np.sin(X*2*np.pi*(5/512))+np.sin(Y*2*np.pi*(40/512))+np.sin(2*np.pi*(X+Y)*(2/512))

plt.imshow(f_x_y, cmap='gray')
plt.title('F10(x,y) after sampling with dx=10 dy=10 in xy space')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#fft
img_fft = np.fft.fft2(f_x_y)
img_fft_shift = np.fft.fftshift(img_fft)
plt.imshow(np.log10(1 + np.abs(img_fft_shift)), cmap='gray')
plt.title('F10(x,y) after sampling with dx=10 dy=10 in frequency space')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#3.e
monkey = cv2.imread('../given_data/Mandrill.jpg') #open image
gray_monkey= cv2.cvtColor(monkey, cv2.COLOR_BGR2GRAY) #convert to grey
plt.imshow(gray_monkey, cmap='gray')
plt.title('Monkey image in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#fft
monkey_fft = np.fft.fft2(gray_monkey)
monkey_fft_shift = np.fft.fftshift(monkey_fft)
plt.imshow(np.log10(1 + np.abs(monkey_fft_shift)), cmap='gray')
plt.title('Amplitude of 2D-DFT of the gray monkey image')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#3.f
# resize to downsample
new_monkey = np.zeros((128, 128))
new_monkey = new_monkey.astype(np.uint8)
smaller = cv2.resize(gray_monkey, (128, 128))


plt.imshow(smaller, cmap='gray')
plt.title('Monkey’s image after down-sampling')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

# downsampled fft
smaller_fft = np.fft.fft2(smaller)
smaller_fft_shift = np.fft.fftshift(smaller_fft)
plt.imshow(np.log10(1 + np.abs(smaller_fft_shift)), cmap='gray')
plt.title('Amplitude of 2D-DFT of the monkey’s image after down-sampling')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


