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

expectedDir = ('/Users/aoife/Documents/University/Year_5/Mphys/coding/SampleA1_tiff2/')
file_data=open('contact_angle_data_tm_a12.txt','w')

def get_image(image_name):
    
    #Read image and convert to black and white format
    image = Image.open( image_name ).convert("L")
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
    box = (1, 250, (height-1), (width-1))
    region = im_clean_edge.crop(box)


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

def droplet_values_alt(points_x, points_y):
    
    height_droplet=(max(points_y)-min(points_y))+2
    #print(height_droplet_r, "height r")
    width_droplet=(max(points_x)-min(points_x))+2

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
    """
    plt.scatter(points_x_right, points_y_right)
    plt.title("right side of droplet")
    plt.show()
    plt.scatter(points_x_left, points_y_left)
    plt.title("left sidr of dropet")
    plt.show()
    """
    return(height_droplet, base_width, width_droplet, points_x_left, points_x_right, points_y_left, points_y_right)

def one_side_only(points_x, points_y, width_droplet, base_width):
    #cut off image edges, x1, y1 to x2, y2

    max_y_=max(points_y)
    points_y.reverse()
    points_x.reverse()
    angles=[]

    for i in range (10):
        y_diff=abs(points_y[i+1]-points_y[0])
        x_diff=abs(points_x[i+1]-points_x[0])
        #print(x_diff, y_diff)
        if y_diff==0:
            continue
        if x_diff==0:
            continue
        else:
            ang=math.degrees(math.atan(y_diff/x_diff))
            angles.append(ang)
        


    if (width_droplet) >= (base_width+5):
        for i in range(len(angles)):
            angles[i]=180-angles[i]
       

    average_angles=np.mean(angles)
   
    
    return(angles, average_angles, angles[0])    
  
start = timeit.default_timer()

file_list=[]
angle_list=[]
time_list=[]
error_listb=[]

for fileName_relative in sorted(glob.glob(expectedDir+"**/*.tiff",recursive=True)):       

    filename = os.path.basename(fileName_relative)              
    file_list.append(filename)

for i in range(500,788):
    
    image_name =  file_list[i] 
    image_initalise=get_image(expectedDir + image_name)



    points_x, points_y = area_of_intrest(image_initalise) 
    
    
    height_droplet, base_width, width_droplet, points_x_left, points_x_right, points_y_left, points_y_right = droplet_values_alt(points_x, points_y)   
    
    
    
    angles_right, average_angles_right, angle_right =one_side_only(points_x_right, points_y_right, width_droplet, base_width) 
    angles_left, average_angles_left, angle_left =one_side_only(points_x_left, points_y_left, width_droplet, base_width)
    
    
    error_list=[]
    
    for j in range((len(angles_right))):
        error_list.append(angles_right[j])
        
    for j in range((len(angles_left))):   
        error_list.append(angles_left[j])
     
    
    error=stat.stdev(error_list)
    variance=stat.variance(error_list)
    
    print("Angle ", (0.5*(angle_right+angle_left)))
    print("error:", error)
    print("variance",variance)
    print(file_list[i],i)
    
    
    #print("Average angle of 10 points closest to surface right;",average_angles_right," left;", average_angles_left,"total average;", (0.5*(average_angles_right+average_angles_left))) 
    #print("Angle of closest data point to surface right;",angle_right, " left;", angle_left, " average;", (0.5*(angle_right+angle_left)))
    #print("ten data points closest to surface angles right;", angles_right)
    #print("ten data points closest to surface angles left;", angles_left) 
    angle_list.append((0.5*(angle_right+angle_left)))
    time_list.append(i)
    error_listb.append(error)
    file_data.write('%lf %lf %lf\n'%(i, (0.5*(angle_right+angle_left)), error))


plt.errorbar(time_list, angle_list,yerr=error_listb)
#plt.title("change in angle")
plt.xlabel("image over time")
plt.ylabel("angle")
plt.show()

plt.plot(time_list, angle_list)
#plt.title("change in angle")
plt.xlabel("image over time")
plt.ylabel("angle")
plt.show()    
    
stop = timeit.default_timer()
print('Run time: ', stop - start)     
file_data.close()
plt.plot(time_list, angle_list)
plt.title("change in angle")
plt.show()    
    