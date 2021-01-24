import numpy as np 
import cv2
import sys
import os
import matplotlib.pyplot as plt
import skimage
# We assume the input image is in the range [0, 255]
def addGaussianNoise(img, mean, var):
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    ret = img + noise
    ret = np.clip(ret, 0, 255)
    return ret

def addSaltPeperNoise(img, prob_salt, prob_peper):
    assert(prob_salt + prob_peper < 1)
    thre1 = prob_salt
    thre2 = prob_salt + prob_peper
    
    tmp = np.random.random(img.shape)
    ret = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if tmp[i,j] < thre1:
                ret[i,j] = 255
            elif tmp[i,j] < thre2:
                ret[i,j] = 0
    return  ret 


def mediaFilter(img, kernel_size):
    H, W = img.shape
    padding = kernel_size // 2
    tmp_img = np.zeros((H + 2 * padding, W + 2 * padding))
    tmp_img[padding: padding + H, padding: padding + W] = img
    ret = np.zeros(img.shape)
    for h in range(H):
        for w in range(W):
            tmp_h = h+ padding
            tmp_w = w+ padding
            ret[h, w]= np.median(tmp_img[tmp_h: tmp_h + kernel_size, tmp_w: tmp_w + kernel_size])
    return ret

    
def addSinNoise(img, A, u0, v0):
    func = lambda x, y: A * np.cos(u0 * x + v0 * y)
    H, W = img.shape
    r = np.arange(H)
    c = np.arange(W)
    rr, cc = np.meshgrid(r, c, indexing='ij')
    noise = A * np.cos(u0 * rr + v0 * cc)
    ret = img + noise 
    return np.clip(ret, 0, 255), noise

# 1D FFT
def FFT1d(signal):
    L = signal.shape[0]
    if L == 1:
        return signal
    ret = np.zeros(signal.shape, dtype = np.complex)
    even_part = FFT1d(signal[::2])
    odd_part = FFT1d(signal[1::2])

    for i in range(L):
        if i < L/2:
            ret[i] = even_part[i] + odd_part[i] * np.exp(-2j * np.pi * 1.0 * i / L)
        else:
            ret[i] = even_part[i-L//2] - odd_part[i-L//2] * np.exp(-2j * np.pi * 1.0 * (i - L/2) / L)
    return ret

def FFT2d(img):
    H, W = img.shape
    fft_result = np.zeros((H, W), dtype = np.complex)
    for n in range(H):
        curr_line = img[n,:]
        fft_result[n,:] = FFT1d(curr_line)
    for m in range(W):
        curr_col = fft_result[:, m]
        fft_result[:, m] = FFT1d(curr_col)
    return fft_result

def IFFT2d(img):
    H, W = img.shape
    conj_img = img.conjugate()
    ret = FFT2d(conj_img)
    ret = ret / (1.0 * H * W)
    ret = ret.conjugate()
    return ret

# shift the image so that the DC part stays at the center of frequenct spectrum 
def shiftImg(img):
    H, W = img.shape
    ret = np.zeros(img.shape, img.dtype)
    for h in range(H):
        for w in range(W):
            ret[h,w] = img[h,w] * np.power(-1, h+w) 
    return ret

def getNotchFilter1DCol(W, H, u0, v0, D0):

    ret = np.zeros((H, W))
    rr = np.arange(H)
    tmp0 = np.exp(-1.0 * (np.power(rr - u0, 2)) / (2 * D0 * D0))
    tmp1 = np.exp(-1.0 * (np.power(rr - (H - u0), 2)) / (2 * D0 * D0))
    ret[:, v0] = tmp0 + tmp1
    return 1 - ret

# To better visualize the spectrum
def myLog(img):
    MAX = 9999999999
    tmp = np.where(img == 0, MAX, img)
    Min = np.min(tmp)
    tmp = np.where(img == 0, Min, img)
    log_img = np.log(tmp)
    return log_img


import time
def blurImage(img, a=0.1, b=0.1, T=1):
    H, W = img.shape
    tmp0 = shiftImg(img)
    spectrum0 = FFT2d(tmp0)

    F = np.zeros(img.shape, dtype= np.complex)

    # 0.01655s
    r = np.arange(H)
    c = np.arange(W)
    rr, cc = np.meshgrid(r, c, indexing='ij')

    pos = np.pi * ((rr - H//2) * a + (cc - W//2) * b)

    pos_ = np.where(pos == 0, 1, pos)
    F = T / pos_ * np.sin(pos_) * np.exp(-1j * pos_)
    F = np.where(pos ==0, T, F)

    # naive implement (0.5799s)
    # for i in range(H):
    #     for j in range(W):
    #         ii = i - H//2
    #         jj = j - W//2
    #         if (ii * a + jj * b) ==0 :
    #             F[i,j] = T
    #         else:
    #             F[i,j] = (T / (np.pi * (ii * a + jj * b)) ) * np.sin(np.pi * (ii*a + jj*b)) * np.exp(-1j * np.pi * (ii * a + jj * b))
    # print(time.time() - t1)

    G = spectrum0 * F

    idft_result = IFFT2d(G)
    idft_real = idft_result.real
    rebuild_img = shiftImg(idft_real)
    
    return rebuild_img, F

    
def myNormalize(img):
    min_num = img.min()
    max_num = img.max()
    img_ret = ((img - min_num) / (max_num - min_num)) * 255
    return img_ret

    
def wienerFilter(img, F, K):
    H, W = img.shape
    tmp0 = shiftImg(img)
    spectrum0 = FFT2d(tmp0)

    res = 1/F * (np.abs(F) ** 2 / (np.abs(F) ** 2 + K)) * spectrum0

    idft_result = IFFT2d(res)
    idft_real = idft_result.real
    rebuild = shiftImg(idft_real)

    return rebuild


# assignment 05-01
def noiseGenerators():

    img = cv2.imread("DIP3E_CH05_Original_Images/Fig0507(a)(ckt-board-orig).tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('img')

    gaussian_img = addGaussianNoise(img, 0, 1000)
    plt.subplot(132), plt.imshow(gaussian_img, 'gray'), plt.title('gaussian_img')

    salt_peper_img = addSaltPeperNoise(img, 0.1, 0.1)
    plt.subplot(133), plt.imshow(salt_peper_img, 'gray'), plt.title('salt_peper_img')

    plt.show()

# assignment 05-02
def noiseReductionUsingaMedianFilter():
    img = cv2.imread("DIP3E_CH05_Original_Images/Fig0507(a)(ckt-board-orig).tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(231), plt.imshow(img, 'gray'), plt.title('img')

    salt_peper_img = addSaltPeperNoise(img, 0.1, 0.1)
    plt.subplot(232), plt.imshow(salt_peper_img, 'gray'), plt.title('salt_peper_img')

    media_filtered_img = mediaFilter(salt_peper_img, 3)
    plt.subplot(233), plt.imshow(media_filtered_img, 'gray'), plt.title('media_filtered_img')

    media_filtered_img = mediaFilter(media_filtered_img, 3)
    plt.subplot(234), plt.imshow(media_filtered_img, 'gray'), plt.title('media_filtered_img')

    plt.show()

# assignment 05-03
def periodicNoiseReductionUsingaNotchFilter():
    img = cv2.imread("DIP3E_CH05_Original_Images/Fig0526(a)(original_DIP).tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    tmp0 = shiftImg(img)
    spectrum0 = FFT2d(tmp0)
    log_spectrum_ori = 1 + myLog(np.abs(spectrum0))

    plt.subplot(241), plt.imshow(img, 'gray'), plt.title('ori_img')
    plt.subplot(245), plt.imshow(log_spectrum_ori, 'gray'), plt.title('spec_ori')

    H, W = img.shape
    print(H, W)
    sin_noise_img, noise= addSinNoise(img, 50, H/2, 0)


    print(sin_noise_img.shape)
    tmp = shiftImg(sin_noise_img)
    spectrum = FFT2d(tmp)
    log_spectrum_noise_img = 1 + myLog(np.abs(spectrum))

    plt.subplot(242), plt.imshow(sin_noise_img, 'gray'), plt.title('sin_noise_img')
    plt.subplot(246), plt.imshow(log_spectrum_noise_img, 'gray'), plt.title('spec_noise_img')

    tmp_noise = shiftImg(noise)
    spectrum_noise = FFT2d(tmp_noise)
    log_spectrum_noise = 1 + myLog(np.abs(spectrum_noise))

    # directly set the spectrum to zero
    # filter = np.ones(img.shape)
    # filter[200:240, 128] = filter[10:50, 128]= 0

    # gaussian filter
    filter = getNotchFilter1DCol(W, H, 223, 128, 10)

    plt.subplot(243), plt.imshow(filter, 'gray'), plt.title('filter')
    plt.subplot(247), plt.imshow(log_spectrum_noise, 'gray'), plt.title('spec_noise')

    filtered_spectrum = filter * spectrum
    log_filtered_spectrum = 1 + myLog(np.abs(filtered_spectrum))
    idft_result = IFFT2d(filtered_spectrum)
    idft_real = idft_result.real
    rebuild_img = shiftImg(idft_real)
    plt.subplot(244), plt.imshow(rebuild_img, 'gray'), plt.title('rebuild_img')
    plt.subplot(248), plt.imshow(log_filtered_spectrum, 'gray'), plt.title('log_filtered_spectrum')
    plt.show()


# assignment 05-04
def parametricWienerFilter():
    img = cv2.imread("DIP3E_CH05_Original_Images/Fig0526(a)(original_DIP).tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    print(img.shape)

    plt.subplot(221), plt.imshow(img, 'gray'), plt.title('img')
    blured_img, F= blurImage(img)
    blured_img = myNormalize(blured_img)
    plt.subplot(222), plt.imshow(blured_img, 'gray'), plt.title('blured_img')

    blured_noise_img = myNormalize(addGaussianNoise(blured_img, 0, 0.0001))
    # blured_noise_img = skimage.util.random_noise(blured_img, mode='gaussian', clip=True, mean=0, var = 10)
    plt.subplot(223), plt.imshow(blured_noise_img, 'gray'), plt.title('blured_noise_img')

    wiener_filtered_img = wienerFilter(blured_noise_img, F, 2)
    # idft_result = IFFT2d(F)
    # idft_real = idft_result.real
    # rebuild_kernel = shiftImg(idft_real)
    # plt.subplot(224), plt.imshow(wiener_filtered_img, 'gray'), plt.title('wiener_filtered_img')
    wiener_filtered_img = wienerFilter(blured_noise_img, F, 0.001)
    plt.subplot(224), plt.imshow(wiener_filtered_img, 'gray'), plt.title('wiener_filtered_img')

    plt.show()


img = cv2.imread("DIP3E_CH05_Original_Images/Fig0525(a)(aerial_view_no_turb).tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
W, H = img.shape
gamma = 1e-5
step = 1e-6
a = 0.7
var = 1e-5
mean = 0


F = np.zeros(img.shape)


r = np.arange(H)
c = np.arange(W)
rr, cc = np.meshgrid(r, c, indexing='ij')

rr = rr - H // 2
cc = cc - W // 2

img_ = shiftImg(img)
G0 = FFT2d(img_)

F = np.exp(-1 * 0.0025 * np.power((rr ** 2 + cc ** 2 ), 5/6))
G_ = F * G0


turb_img = IFFT2d(G_)
turb_img = turb_img.real
turb_img = shiftImg(turb_img)
turb_img = myNormalize(turb_img)
plt.subplot(221), plt.imshow(turb_img, 'gray'), plt.title('turb_img')

turb_noise_img = myNormalize(addGaussianNoise(turb_img, mean, var))
plt.subplot(222), plt.imshow(turb_noise_img, 'gray'), plt.title('turb_noise_img')

p = np.zeros(img.shape)
p[0:3, 0:3] = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])

p = shiftImg(p)
P = FFT2d(p)

img_ = shiftImg(turb_noise_img)
G = FFT2d(img_)

noise_2 = H * W * (var  + mean ** 2)
cnt = 0

while True:
    cnt+=1
    F_hat = G *np.conj(F) / ((np.abs(F)) ** 2 + gamma * ((np.abs(P)) ** 2))
    R = G - F * F_hat

    R_ = IFFT2d(R)
    R_ = R_.real
    r_ = shiftImg(R_)

    r_2 = np.sum(r_ ** 2)
    print("r_2", r_2)
    print("noise_2", noise_2)
    print('(' + str(cnt)+') ', np.abs(r_2 - noise_2))
    if r_2 < noise_2 -a :
        gamma += step
    elif r_2 > noise_2 + a:
        gamma -= step
    else:
        break

    f_hat = IFFT2d(F_hat)
    f_hat = f_hat.real
    f_hat = shiftImg(f_hat)
    f_hat = myNormalize(f_hat)
    cv2.imwrite(str(cnt) + "imgs/_res.png", f_hat)

plt.subplot(223), plt.imshow(f_hat, 'gray'), plt.title('f_hat')

plt.show()



