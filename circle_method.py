import numpy as np
import math
import matplotlib.pyplot as plt 
from scipy  import optimize
from numpy import *
import statistics as stat
from scipy.stats import chisquare

from PIL import Image
from PIL import ImageFilter
import timeit 

def get_image(image_name):
    
    #Read image and convert to black and white format
    image = Image.open( image_name ).convert("L")
    #image.show()
    
    
    
    #finds edges of the objects in the image.
    im_edge = image.filter(ImageFilter.FIND_EDGES)
    
    image_array = np.asarray(im_edge)
    width, height = image_array.shape
    
    listh=[]
    for i in range (width):
        for j in range (height):
           if image_array[i,j]>=0: 
               listh.append(image_array[i,j]) 
    
    plt.hist(listh, bins=30)
    plt.savefig("histogram.png")
    plt.show()    
    plt.close()

    #threshold image to clear noise
    threshold = 180       #180
    im_clean_edge = im_edge.point(lambda p: p > threshold and 225) 
    im_clean_edge.show()
    
    return(im_clean_edge)

def area_of_intrest(im_clean_edge):

    #cut off image edges
    image_array = np.asarray(im_clean_edge)
    width, height = image_array.shape
    box =  (1, 250, (height-1), (width-1)) #(200, 250, 450, (450)) 
    region = im_clean_edge.crop(box)
    #region.show()

    #create array of image pixels, 225= white edge pixel, 0 = black background pixel
    image_array = np.asarray(region)
    position_pixel=np.argwhere(image_array == 225)


    points_x=[]
    points_y=[]
    points_y_r=[]

    #loop over entire array - 0 is x direction, 1 is y direction for array size.-3 due to box cutting off edges
    for i in range ((region.size[1])-1):
        for j in range ((region.size[0])-1):
            if image_array[i][j]==225:
                points_x.append(j)
                points_y.append(i)
                
                
    #flipping y values so graph is not upside down
    max_y=max(points_y)
    for x in range(len(points_y)):
        points_y_r.append(max_y-points_y[x])

    return(points_x, points_y_r)

def droplet_values(points_x, points_y):
    
    height_droplet=(max(points_y)-min(points_y))+2
    #print(height_droplet_r, "height r")

    #find droplets base width, split into two parts left and right halves to do this
    points_x_left=[]
    points_x_right=[]
    points_y_left=[]
    points_y_right=[]

    for i in range(len(points_x)):
        if points_y[i]==max(points_y):
            y_max_xc=points_x[i]
    #print(y_max_xc,"x coordinate")

    for i in range(len(points_x)):
        if points_x[i]<=y_max_xc:
            points_x_left.append(points_x[i])
            points_y_left.append(points_y[i])
        else:
            points_x_right.append(points_x[i])
            points_y_right.append(points_y[i])


    for i in range(len(points_x_left)):
        if points_y_left[i]==min(points_y_left):
            R_p_left=points_x_left[i]
        
    
    for i in range(len(points_x_right)):
        if points_y_right[i]==min(points_y_right):
            R_p_right=points_x_right[i]
        
 
    base_width=(R_p_right-R_p_left)+2
    #print(base_width, "Base width")
    
    return(height_droplet, base_width)

def calculate_circle_fit(points_x, points_y, x, y):
    
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m
    #estimate centre
    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)
    #fit circle
    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)
    return(R_2, xc_2, yc_2)

def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return sqrt((x-xc)**2 + (y-yc)**2)

def f_2(c):
    """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()

def angle_circle_fit(height_droplet, R_2):
    
    if height_droplet <=R_2:
        h=R_2-height_droplet
        theta=math.degrees(math.asin(h/R_2))
        angle=90-theta
    
    else:
        h=height_droplet-R_2
        print(h, R_2, "here here")
        theta=math.degrees(math.asin(h/R_2))
        angle=90+theta

    return(angle)

def angle_circle_fit_alternative(height_droplet, R_2, base_width):
    #based on equation from 
    
    x=base_width/2
    
    if height_droplet <= R_2:
        theta=math.degrees(math.asin(x/R_2))
        angle=90-theta
    
    else:
        theta=math.degrees(math.asin(x/R_2))
        angle=90+theta

    return(angle)

def errors(points_x, points_y,circle_height, xc_2, yc_2):
    diff_list=[]
    for i in range(len(points_x)):
      x=points_x[i]
      y=points_y[i]
      distance=math.sqrt(((xc_2-x)**2)+((yc_2-y)**2))
      diff=abs(distance-circle_height)
      diff_list.append(diff)
      
      
    error=sum(diff_list)/(len(diff_list))
    return(error)

def chi_fit(points_x, points_y,circle_height, xc_2, yc_2):
    
    distance_list=[]
    radius_list=[]
    
    for i in range(len(points_x)):
        x=points_x[i]
        y=points_y[i]
        distance=math.sqrt(((xc_2-x)**2)+((yc_2-y)**2))
        distance_list.append(distance)
        radius_list.append(circle_height)
        #print(distance, circle_height)
        
    chi,p=chisquare(f_obs=distance_list,f_exp=radius_list)
    print(chi, "chi",p)  
    print(chi/(len(points_x)),"chi2")
        
    return(chi)
    
      
 

start = timeit.default_timer()
#set file name, then set up image
file=('droplet_1.JPEG')
image_name=file
image_initalise=get_image(image_name)

#select area of intrest within box, then convert data to an array
#set are of intrest in are_of_intrest method at box selection
#graph the shape of the are of intrest as a scatter plot
points_x, points_y = area_of_intrest(image_initalise)
plt.scatter(points_x, points_y)
plt.title("edge of droplet")
plt.show()

#find values for the base width and the droplets height
height_droplet, base_width = droplet_values(points_x, points_y)

#fit circle to the droplet using only the 10 values around the top point of the circle
x = r_[points_x[-10:]]
y = r_[points_y[-10:]]
circle_height, xc_2, yc_2 = calculate_circle_fit(points_x, points_y, x, y)

circle1 = plt.Circle((xc_2, yc_2), circle_height, color='b', fill=False)
ax = plt.gca()
ax.cla() 
ax.set_xlim((300, 430))
ax.set_ylim((0, 150))
ax.scatter(points_x, points_y, color='orange')
ax.add_artist(circle1)
#plt.title("circle fit of droplet")
plt.show()

#find the angle
droplet_angle=angle_circle_fit(height_droplet, circle_height)
print("angle of droplet;",droplet_angle)

droplet_angle_alt=angle_circle_fit_alternative(height_droplet, circle_height, base_width)
print("alternative equation droplet angle;", droplet_angle_alt)

error=errors(points_x, points_y,circle_height, xc_2, yc_2)
print('error', error)

chi=chi_fit(points_x, points_y,circle_height, xc_2, yc_2)

stop = timeit.default_timer()
print('Time: ', stop - start) 