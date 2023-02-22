import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#2.a
img_parrot = cv2.imread('../given_data/parrot.png') #open image
gray_parrot = cv2.cvtColor(img_parrot, cv2.COLOR_BGR2GRAY) #convert to gray
size_parrot = gray_parrot.shape
yours = cv2.imread('../my_data/yours.jpg') #open image
ours_resize = cv2.resize(yours, (size_parrot[0],size_parrot[1]))    #resize
gray_ours = cv2.cvtColor(ours_resize, cv2.COLOR_BGR2GRAY)  #convert to gray

plt.imshow(gray_ours, cmap='gray')
plt.title('our selfie')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

plt.imshow(gray_parrot, cmap='gray')
plt.title('parrot')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()
#2.b
parrot_fft = np.fft.fft2(gray_parrot) ##2d fft
parrot_fft_shift = np.fft.fftshift(parrot_fft)
parrot_amp=np.abs(parrot_fft_shift) #amplitude
parrot_phase=np.angle(parrot_fft_shift) #phase

plt.imshow(parrot_amp, cmap='gray') #show image
plt.title('parrot amplitude')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

plt.imshow(parrot_phase, cmap='gray') #show image
plt.title('parrot phase')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

ours_fft = np.fft.fft2(gray_ours)  ##2d fft
ours_fft_shift = np.fft.fftshift(ours_fft)
ours_amp=np.abs(ours_fft_shift)
ours_phase=np.angle(ours_fft_shift)
plt.imshow(ours_amp, cmap='gray') #show image
plt.title('our amplitude')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

plt.imshow(ours_phase, cmap='gray') #show image
plt.title('our phase')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


#2.c
combination_1 = np.multiply(ours_amp, np.exp(1j * parrot_phase))  #mix our amplitude with parrot phase
combination_1_shift = np.fft.ifftshift(combination_1)
first_img=abs(np.fft.ifft2(combination_1_shift))

plt.imshow(first_img, cmap='gray') #show image
plt.title('our amplitude and parrot phase')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


combination_2 = np.multiply(parrot_amp, np.exp(1j * ours_phase)) #mix our phase with parrot amplitude
combination_2_shift = np.fft.ifftshift(combination_2)
second_img=abs(np.fft.ifft2(combination_2_shift))
plt.imshow(second_img, cmap='gray') #show image
plt.title('our phase and parrot amplitude')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

#2.d
#create random phase
max_phase=max(map(max, ours_phase))
min_phase = min(map(min, ours_phase))
random_phase =np.empty(ours_phase.shape)
for i in range(0,ours_phase.shape[0]):
    for j in range(0,ours_phase.shape[1]):
        random_phase[i][j]= np.random.uniform(min_phase, max_phase)

#create our combination with oure amp and random phase
combination_3 = np.multiply(ours_amp, np.exp(1j * random_phase))
combination_3_shift = np.fft.ifftshift(combination_3)
third_img=abs(np.fft.ifft2(combination_3_shift))
plt.imshow(third_img, cmap='gray') #show image
plt.title('ours amp and random phase')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


#create random amplitude
max_amp=max(map(max, ours_amp))
min_amp = min(map(min, ours_amp))
random_amp =np.empty(ours_amp.shape)

for i in range(0,ours_amp.shape[0]):
    for j in range(0,ours_amp.shape[1]):
        random_amp[i][j]=np.random.uniform(min_amp, max_amp)

#create our combination with ours phase and random amplirude
combination_4 = np.multiply(random_amp, np.exp(1j * ours_phase))
combination_4_shift = np.fft.ifftshift(combination_4)
fourth_img=abs(np.fft.ifft2(combination_4_shift))
plt.imshow(fourth_img, cmap='gray') #show image
plt.title('ours phase and random amplitude')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


