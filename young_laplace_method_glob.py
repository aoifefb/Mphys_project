import numpy as np
import math
import matplotlib.pyplot as plt 
from scipy  import optimize
from numpy import *
import statistics as stat
import glob 
import os

from PIL import Image
from PIL import ImageFilter
import timeit

density = 1000  #kg/m^3
surface_tension = 0.072  #mN
gravity= 9.80665  #m/s2
delta_angle=0.001 #rads  (10^-4 in behrozzi)
conversion=158400

expectedDir = ('/Users/aoife/Documents/University/Year_5/Mphys/coding/noise/')
file_data=open('contact_angle_data_ylp_noise2.txt','w')

def get_image(image_name):
    
    #Read image and convert to black and white format
    image = Image.open( expectedDir + image_name ).convert("L")
    #image.show()

    #finds edges of the objects in the image.
    im_edge = image.filter(ImageFilter.FIND_EDGES)

    #threshold image to clear noise
    threshold = 180
    im_clean_edge = im_edge.point(lambda p: p > threshold and 225) 
    #im_clean_edge.show()
    
    return(im_clean_edge)

def area_of_intrest(im_clean_edge):

    #cut off image edges
    image_array = np.asarray(im_clean_edge)
    width, height = image_array.shape
    box = (1, 250, (height-1), (width-1)) #second option 250
    region = im_clean_edge.crop(box)


    #create array of image pixels, 225= white edge pixel, 0 = black background pixel
    image_array = np.asarray(region)
    #position_pixel=np.argwhere(image_array == 225)

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
    R_p_right=0
    R_p_left=0
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

def young_laplace_integration(r_not, height, delta_angle, surface_tension,density, gravity, conversion):
    x=0
    y=0
    theta=0
    x_list=[]
    y_list=[]
    x_dashed=r_not
    y_dashed=0
    angle_list=[]
    angle=0
    
    
    surface_tension=surface_tension*conversion
    gravity=gravity*conversion
    density=density/(conversion**3)
    
    
    for i in range(10000):
        
        x_list.append(x)
        y_list.append(y)
        
        x=x+(x_dashed*(delta_angle/2))
        y=y+(y_dashed*(delta_angle/2))
        theta=theta+(delta_angle/2)
    
        base=((2*surface_tension/r_not)+(density*gravity*y)-((surface_tension/x)*math.sin(theta)))
        x_dashed=(surface_tension*math.cos(theta))/base 
        y_dashed=(surface_tension*math.sin(theta))/base 
    
        x=x_list[i]+(delta_angle*x_dashed)
        y=y_list[i]+(delta_angle*y_dashed)   
        theta=theta+(delta_angle/2)
        
        base=((2*surface_tension/r_not)+(density*gravity*y)-((surface_tension/x)*math.sin(theta)))
        x_dashed=(surface_tension*math.cos(theta))/base 
        y_dashed=(surface_tension*math.sin(theta))/base 
        
        if round(y)==height:
            angle_list.append(math.degrees(theta))
            #print("here", i)
        
    angle=sum(angle_list)/len(angle_list)
    #print(angle_list)
    if len(angle_list)<=1:
        error=1
        variance=1
    else:
        error=stat.stdev(angle_list)
        variance=stat.variance(angle_list)
    return(angle,error, variance)



start = timeit.default_timer()
#set file name, then set up image


file_list=[]
angle_list=[]
time_list=[]
error_list=[]
for fileName_relative in sorted(glob.glob(expectedDir+"**/*.tiff",recursive=True)):       

    filename = os.path.basename(fileName_relative)              
    file_list.append(filename)

for i in range (len(file_list)):

    image_name =  file_list[i] 
    image_initalise=get_image( image_name)

    #select area of intrest within box, then convert data to an array
    #set are of intrest in are_of_intrest method at box selection
    #graph the shape of the are of intrest as a scatter plot
    points_x, points_y = area_of_intrest(image_initalise)
    
    if len(points_x)>=2 and len(points_y)>=2:

        #find values for the base width and the droplets height
        height_droplet, base_width = droplet_values(points_x, points_y)
        #print(height_droplet, "droplet height")
        
        
        #fit circle to the droplet using only the 10 values around the top point of the circle
        x = r_[points_x[-10:]]
        y = r_[points_y[-10:]]
        circle_height, xc_2, yc_2 = calculate_circle_fit(points_x, points_y, x, y)
        
        angle,error,var = young_laplace_integration(circle_height, height_droplet, delta_angle, surface_tension, density, gravity, conversion)
        #chi=chi_fit2(x_list, y_list,circle_height, xc_2, yc_2, conversion,list_max, points_x_right, points_y_right)
        
        print("angle;", angle)
        print("standard deviation:", error)
        print("Variance", var)
       
        print(file_list[i],i)
        
        
       
        
        angle_list.append(angle)
        time_list.append(i)
        error_list.append(error)
        file_data.write('%lf  %lf  %lf\n'%( angle, error, var))

stop = timeit.default_timer()
file_data.close()
print('Run time: ', stop - start) 

plt.errorbar(time_list, angle_list,yerr=error_list)
#plt.title("change in angle")
plt.xlabel("image over time")
plt.ylabel("angle")
plt.show()

plt.plot(time_list, angle_list)
#plt.title("change in angle")
plt.xlabel("image over time")
plt.ylabel("angle")
plt.show()