import cv2 as cv
import numpy as np

image = cv.imread('lena256.bmp')
original_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
h, w = original_image.shape
kernel_size = 3
kernel = cv.getGaussianKernel(kernel_size, 2)
kernel = kernel @ kernel.T
output = cv.filter2D(original_image, -1, kernel)


# print(original_image[1].shape)
# image =  np.array(original_image).flatten()
# length = image.shape[0]
# print(image.shape)
# kernel_size = 10
# kernel = cv.getGaussianKernel(kernel_size, 2)
# kernel = kernel @ kernel.T
# stride = 10

# convold_kernel = np.array([])
# for i in range(0, int((h-kernel_size)), stride):
#     for j in range(0, int((w-kernel_size)), stride):
#         template = np.zeros((h, w))
#         template[i:i+kernel_size, j:j+kernel_size] = kernel
#         template = np.array(template).flatten()
        
#         if len(convold_kernel) == 0:
#             convold_kernel = template
#         else:
#             convold_kernel = np.vstack((convold_kernel, template))


# output = image @ convold_kernel.T
# size = int(np.sqrt(len(output)))
# output = output.reshape((size,size))
# output = np.array(output, dtype=np.uint8)
# print(output.shape)

# recover = image @ convold_kernel.T @ convold_kernel
# recover = recover.reshape((h, w))
# recover = np.array(recover, dtype=np.uint8)
# print(recover.shape)




# print(kernel)
# output = cv.filter2D(image, -1, kernel)
# print(image.shape, output.shape, recover.shape)
cv.imwrite('lena256_blurred.bmp', output)
cv.imshow('Original Image', original_image)
cv.imshow('Convolution Result', output)
# cv.imshow('recover Result', recover)

cv.waitKey(0)
cv.destroyAllWindows()