import numpy as np
from PIL import Image

image_name=('droplet_1.JPEG')
image = Image.open( image_name ).convert("L")
#image.show()

img = np.asarray(image)
row, col = img.shape
        #np.interp(img.min(),img.max(),(0,1))

mean = 0   # some constant
std = 10    # some constant (standard deviation)
noisy_img = img + np.random.normal(mean, std, img.shape)
noisy_img_clipped = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise

im2=Image.fromarray(noisy_img_clipped)
im2.show() 



"""

This works
noisy_img = img + np.random.normal(mean, std, img.shape)
noisy_img_clipped = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise
im2=Image.fromarray(noisy_img_clipped)
im2.show()



import skimage
import matplotlib.pyplot as plt
img_path="https://i.guim.co.uk/img/media/4ddba561156645952502f7241bd1a64abd0e48a3/0_1251_3712_2225/master/3712.jpg?width=1920&quality=85&auto=format&fit=max&s=1280341b186f8352416517fc997cd7da"
img = skimage.io.imread(img_path)/255.0

def plotnoise(img, mode, r, c, i):
    plt.subplot(r,c,i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode)
        plt.imshow(gimg)
    else:
        plt.imshow(img)
    plt.title(mode)
    plt.axis("off")

plt.figure(figsize=(18,24))
r=4
c=2
plotnoise(img, "gaussian", r,c,1)
plotnoise(img, "localvar", r,c,2)
plotnoise(img, "poisson", r,c,3)
plotnoise(img, "salt", r,c,4)
plotnoise(img, "pepper", r,c,5)
plotnoise(img, "s&p", r,c,6)
plotnoise(img, "speckle", r,c,7)
plotnoise(img, None, r,c,8)
plt.show()


normal_array = image_array.astype(float)/1.0


print(normal_array)
mean = 0
var = 0.01
sigma = var**0.5

#im_2=np.interp(image_array.min(),image_array.max(),(0,1))
#print(im_2)

gauss = 0.1*np.random.normal(mean,sigma,(row,col))
#print(gauss)
gauss = gauss.reshape(row,col)
#print(gauss)
noisy = normal_array + gauss

min_noisy=np.amin(noisy)
noisy_shift=noisy-min_noisy

max_shift=np.amax(noisy_shift)
noisy_scale=(1.0/max_shift)*noisy_shift

print(np.amin(noisy_scale),np.amax(noisy_scale))
#im_2=np.interp(np.amin(noisy),np.amax(noisy),(0,1))
#print(noisy_scale)

#noisy_scale=(noisy_scale*255.0)

im2=Image.fromarray(normal_array, 'L')
im2.show() 

im=Image.fromarray(noisy_scale, 'L')
im.show() 


#print(im_2)            

print(noisy)
gauss = gauss.reshape(row,col,ch)
im=Image.fromarray(noisy)
im.show()
"""

