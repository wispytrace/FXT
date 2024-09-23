import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from scipy.io import loadmat
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


faces = loadmat('resources\image_recover\olivettifaces.mat')['faces']

image = faces[:,12].reshape(64, 64)
print(image.shape)
image_vector = image.flatten()[:, np.newaxis]  
kernel = np.array([[1, 2, 1],  
                   [2, 4, 2],  
                   [1, 2, 1]]) / 16

output_size = (image.shape[0] - kernel.shape[0] + 1) * (image.shape[1] - kernel.shape[1] + 1)  
blur_matrix = np.zeros((output_size, image.size))

for i in range(image.shape[0] - kernel.shape[0] + 1):  
    for j in range(image.shape[1] - kernel.shape[1] + 1):  
        row_index = i * (image.shape[1] - kernel.shape[1] + 1) + j  
        col_index = 0  
        for ki in range(kernel.shape[0]):  
            for kj in range(kernel.shape[1]):  
                if i + ki < image.shape[0] and j + kj < image.shape[1]:  
                    blur_matrix[row_index, (i + ki) * image.shape[1] + (j + kj)] = kernel[ki, kj]  
                col_index += 1  
                
blurred_image_vector = np.dot(blur_matrix, image_vector)  
blurred_image = blurred_image_vector.reshape((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1))

plt.plot(image)
# plt.show()
plt.savefig("face.png")


PSNR = peak_signal_noise_ratio(image, blurred_image)
print('PSNR: ', PSNR)
SSIM = structural_similarity(image, blurred_image)
print('SSIM: ', SSIM)

print("Original Image:")  
print(image)  
print("\nBlurred Image:")
print(blurred_image)
plt.imshow(blurred_image)
plt.show()