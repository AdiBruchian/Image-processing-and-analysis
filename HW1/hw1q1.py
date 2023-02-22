import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


#1.a
img = cv2.imread('../my_data/building.jpg') #open image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray-scale
plt.imshow(img_gray, cmap='gray')
plt.title('image of building in gray-scale')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


#1.b
img_gray_fft = np.fft.fft2(img_gray) #creating the 2D-DFT of the image
img_gray_fftshift = np.fft.fftshift(img_gray_fft) #bring the low frequencies to the center of the transform image
img_fftshift_amp = np.log(1 + np.abs(img_gray_fftshift))
plt.imshow(img_fftshift_amp, cmap='gray')
plt.title('Amplitude of 2D-DFT of the gray building image')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


#1.c
#this function calculate the abs of inverse transform
def ifft_abs(img_fft):
 img_ishift = np.fft.ifftshift(img_fft)
 img_ifft = np.fft.ifft2(img_ishift)
 img_ifft_abs = np.abs(img_ifft)
 return img_ifft_abs

strip_size = 0.02 #define the lowest 2% frequencies
hieght, width = img_gray_fftshift.shape

#creating lowest 2% frequencies - l direction
l_slice = slice(int((width/2)-(width*(strip_size/2))),int((width/2)+(width*(strip_size/2))+1)) #create vertical slice
l_strip = np.zeros_like(img_gray_fftshift)
l_strip[:,l_slice] = img_gray_fftshift[:,l_slice]
l_strip_abs = np.log(1+np.abs(l_strip))
plt.imshow(l_strip_abs, cmap='gray')
plt.title('vertical LPF - The lowest 2% frequencies in the l direction')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

l_strip_ifft_abs = ifft_abs(l_strip)
plt.imshow(l_strip_ifft_abs, cmap='gray')
plt.title('Inverse transform image after vertical LPF')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


#creating lowest 2% frequencies - k direction
k_slice = slice(int((hieght/2)-((strip_size/2)*hieght)),int((hieght/2)+((strip_size/2)*hieght)+1))#create horizontal slice
k_strip = np.zeros_like(img_gray_fftshift)
k_strip[k_slice,:] = img_gray_fftshift[k_slice,:]
k_strip_abs = np.log(1+np.abs(k_strip))
plt.imshow(k_strip_abs, cmap='gray')
plt.title('horizontal LPF - The lowest 2% frequencies in the k direction')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

k_strip_ifft_abs = ifft_abs(k_strip)
plt.imshow(k_strip_ifft_abs, cmap='gray')
plt.title('Inverse transform image after horizontal LPF')
plt.xlabel("l")
plt.ylabel("k")
plt.show()

#creating lowest 2% frequencies - k and l directions
kl_strip = np.zeros_like(img_gray_fftshift)
kl_strip[:,l_slice] = img_gray_fftshift[:,l_slice]
kl_strip[k_slice,:] = img_gray_fftshift[k_slice,:]
kl_strip_abs = np.log(1+np.abs(kl_strip))
plt.imshow(kl_strip_abs, cmap='gray')
plt.title('vertical and horizontal LPF - The lowest 2% frequencies in k and l directions')
plt.xlabel("l")
plt.ylabel("k")
plt.show()


kl_strip_ifft_abs = ifft_abs(kl_strip)
plt.imshow(kl_strip_ifft_abs, cmap='gray')
plt.title('Inverse transform image after k and l LPF')
plt.xlabel("l")
plt.ylabel("k")
plt.show()

#1.d
def max_freq_filtering(fshift, precentege):
 """
 Reconstruct an image using only its maximal amplitude frequencies.
 :param fshift: The fft of an image, **after fftshift** -
 complex float ndarray of size [H x W].
 :param precentege: the wanted precentege of maximal frequencies.
 :return:
 fMaxFreq: The filtered frequency domain result -
 complex float ndarray of size [H x W].
 imgMaxFreq: The filtered image - real float ndarray of size [H x W].
 """
 # ====== YOUR CODE: ======

 unwanted_elements = (fshift.shape[0])*(fshift.shape[1])*(1-precentege/100)
 fshift_array = fshift.reshape(-1) #change shape to ease finding the maximal frequencies
 fshift_array_abs = np.log(1 + np.abs(fshift_array))
 fshift_abs_sorted = np.argpartition(fshift_array_abs, int(unwanted_elements))
 fshift_array[fshift_abs_sorted[:int(unwanted_elements)]] = 0 #indices of the [100-percentage] elements
 fMaxFreq =fshift_array.reshape(fshift.shape)
 imgMaxFreq = np.log(1 + np.abs(fMaxFreq))
 # ========================
 return fMaxFreq, imgMaxFreq


fMaxFreq_10, imgMaxFreq_10 = max_freq_filtering(img_gray_fftshift, 10)
plt.imshow(imgMaxFreq_10, cmap='gray')
plt.title('Maximal 10% frequencies')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

fMaxFreq_10_ifft_abs = ifft_abs(fMaxFreq_10)
plt.imshow(fMaxFreq_10_ifft_abs, cmap='gray')
plt.title('Inverse transform image after 10% max filter')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


#1.e
fMaxFreq_4, imgMaxFreq_4 = max_freq_filtering(img_gray_fftshift, 0.1)
plt.imshow(imgMaxFreq_4, cmap='gray')
plt.title('Maximal 4% frequencies')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

fMaxFreq_4_ifft_abs = ifft_abs(fMaxFreq_4)
plt.imshow(fMaxFreq_4_ifft_abs, cmap='gray')
plt.title('Inverse transform image after 4% max filter')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


#1.f
mse = np.zeros(100)
percentage = range(1, 101)
for p in percentage:
    filtered_freq, filtered_freq_img = max_freq_filtering(img_gray_fftshift, p)
    reconst_img = ifft_abs(filtered_freq)
    mse[p-1] = np.mean((img_gray- reconst_img)**2)

fig = plt.figure(figsize=(8, 5)) # create a figure
ax = fig.add_subplot(1, 1 ,1) # create a subplot of certain size
ax.plot(percentage, mse)
ax.set_xlabel('percentage [%]')
ax.set_ylabel("MSE")
ax.set_title("MSE as function of the percentage of maximal frequencies")
ax.grid()
plt.show()

