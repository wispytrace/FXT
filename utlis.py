import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

# img1 = cv2.imread('output/image_recover/0/result40.bmp', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('resources/image_recover/original/lena256.bmp', cv2.IMREAD_GRAYSCALE)
# img3 = cv2.imread('resources/image_recover/blurred/lena256_blurred.bmp', cv2.IMREAD_GRAYSCALE)
 

# PSNR = peak_signal_noise_ratio(img2, img1)
# print('PSNR: ', PSNR)
# SSIM = structural_similarity(img2, img1)
# print('SSIM: ', SSIM)

# PSNR = peak_signal_noise_ratio(img2, img3)
# print('PSNR: ', PSNR)
# SSIM = structural_similarity(img2, img3)
# print('SSIM: ', SSIM)
matrix = np.matrix([[2.4, 0.2, 0.2, 0.2, 0.2],
                    [0.2, 2.4, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 2.4, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 2.4, 0.2],
                    [0.2, 0.2, 0.2, 0.2, 2.4],])

print(np.linalg.eigvals(matrix))