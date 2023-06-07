import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv


# 加密函数
def encrypt(img, key):
    # 将图像转换为一维数组
    img_flat = img.flatten()
    # 将密钥复制到与图像数组相同的形状
    key_flat = np.resize(key, img_flat.shape)
    # 对图像数组和密钥数组进行异或运算
    encrypted_flat = np.bitwise_xor(img_flat, key_flat)
    # 将加密后的数组转换回图像形状
    encrypted_img = encrypted_flat.reshape(img.shape)
    encrypted_img = encrypted_img.astype(np.uint8)
    return encrypted_img


### 将水印加在图片上
# 读取原图和水印图并傅里叶变换
img = cv2.imread('image.png', 0)
watermark = cv2.imread('watermark.jpeg', 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dftshift = np.fft.fftshift(dft)
# 加密变换
with open('matrix.csv', 'r') as file:
    reader = csv.reader(file)
    key = [[int(num) for num in row] for row in reader]
# key = np.random.randint(0, 256, size=img.shape)
watermark = encrypt(watermark, key)
dft_water = cv2.dft(np.float32(watermark), flags=cv2.DFT_COMPLEX_OUTPUT)
dftshift_water = np.fft.fftshift(dft_water)

# 频域信息叠加
alpha = 0.15
dftadd = dftshift + alpha * dftshift_water

# 傅里叶逆变换
ishift = np.fft.ifftshift(dftshift)
iimg = cv2.idft(ishift)
res2 = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

iadd = np.fft.ifftshift(dftadd)
iimg_add = cv2.idft(iadd)
res33 = cv2.magnitude(iimg_add[:, :, 0], iimg_add[:, :, 1])
# 归一化
cv2.normalize(res33, res33, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("watermarked.png", res33)


### 根据加了水印的图片和原图片, 得到水印
re_img = cv2.imread('image.png', 0)
re_watermarked = cv2.imread('watermarked.png', 0)

re_dft = cv2.dft(np.float32(re_img), flags=cv2.DFT_COMPLEX_OUTPUT)
re_dftshift = np.fft.fftshift(re_dft)
re_dft_watermarked = cv2.dft(np.float32(re_watermarked), flags=cv2.DFT_COMPLEX_OUTPUT)
re_dftshift_watermarked = np.fft.fftshift(re_dft_watermarked)

re_water = (1 / alpha) * (re_dftshift_watermarked - re_dftshift)
ire_en_watermarked = np.fft.ifftshift(re_water)
iimg_re_en_watermarked = cv2.idft(ire_en_watermarked)
img_re_en_watermarked = cv2.magnitude(iimg_re_en_watermarked[:, :, 0], iimg_add[:, :, 1])
img_re_en_watermarked = np.uint8(img_re_en_watermarked /(re_img.shape[0]*re_img.shape[1]))
img_re_watermarked = 255 - encrypt(img_re_en_watermarked, key)

# 显示图像
plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(132), plt.imshow(res33, 'gray'), plt.title('watermarked Image')
plt.axis('off')
plt.subplot(133), plt.imshow(img_re_watermarked, 'gray'), plt.title('watermark')
plt.axis('off')
plt.show()
