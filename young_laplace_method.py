import numpy as np
import math
import matplotlib.pyplot as plt 
from scipy  import optimize
from numpy import *
import statistics as stat
import glob 
from scipy.stats import chisquare

from PIL import Image
from PIL import ImageFilter
import timeit

density = 1000  #kg/m^3
surface_tension = 0.072  #mN
gravity= 9.80665  #m/s2
delta_angle=0.001 #rads  (10^-4 in behrozzi)
conversion=158400 


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
    box = (1, 250, (height-1), (width-1)) #second option 250
    region = im_clean_edge.crop(box)
    #region.show()

    #create array of image pixels, 225= white edge pixel, 0 = black background pixel
    image_array = np.asarray(region)
    position_pixel=np.argwhere(image_array == 225)

    histl=[]
    points_x=[]
    points_y=[]
    points_y_r=[]

    #loop over entire array - 0 is x direction, 1 is y direction for array size.-3 due to box cutting off edges
    for i in range ((region.size[1])-1):
        for j in range ((region.size[0])-1):
            histl.append(image_array[i,j])
            if image_array[i][j]==225:
                points_x.append(j)
                points_y.append(i)
                
    plt.hist(histl, bins=30)
    
    plt.show()    
    plt.close()            
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
    plt.scatter(points_x_right, points_y_right)
    plt.title("right side of droplet")
    plt.show()
    plt.scatter(points_x_left, points_y_left)
    plt.title("left sidr of dropet")
    plt.show()
    return(height_droplet, base_width, width_droplet, points_x_left, points_x_right, points_y_left, points_y_right)

def droplet_values(points_x, points_y):
    
    height_droplet=(max(points_y)-min(points_y))+2
    #print(height_droplet_r, "height r")

    #find droplets base width, split into two parts left and right halves to do this
    points_x_left=[]
    points_x_right=[]
    points_y_left=[]
    points_y_right=[]
    R_p_right=0
    R_p_left=0

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
    print(base_width, "Base width")
    
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

def young_laplace_integration(r_not, height, delta_angle, surface_tension,density, gravity, conversion, xc_2,yc_2):
    x=0
    y=0
    theta=0
    x_list=[]
    y_list=[]
    x_list_actual=[]
    y_list_actual=[]
    x_dashed=r_not
    y_dashed=0
    angle_list=[]
    angle=0
    h=0
    
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
            if h==0:
                y_max=i
            h=h+1
            
        
    angle=sum(angle_list)/len(angle_list)
    #print(angle_list)
    if len(angle_list)>=2:
        error=stat.stdev(angle_list)
        variance=stat.variance(angle_list)
    else:
        error=1
        variance=1
        
    #chi_fit(x_list,y_list,r_not,xc_2,yc_2, conversion)
    #print(y_max)
    
    return(angle,error, variance, x_list ,y_list ,y_max)


def chi_fit(points_x, points_y,circle_height, xc_2, yc_2, conversion,ymax, x_right, y_right):
    
    x_list=[]
    y_list=[]
    
    for i in range(ymax+1):
        x_list.append((points_x[i]))
        y_list.append((points_y[i]-points_y[ymax])*-1)
    
    
    
    x = r_[x_list]
    y = r_[y_list]
    r, xc, yc = calculate_circle_fit(x_list, y_list, x, y)
    f_exp=[r,xc,yc]
    f_obs=[circle_height,xc_2,yc_2]
    
    print(f_exp)
    print(f_obs)
        
    chi,p=chisquare(f_obs=f_obs,f_exp=f_exp)
    print(chi, "chi",p)  
    print(chi/(len(points_x)),"chi2")
    
        
    return(chi)

def chi_fit2(points_x, points_y,circle_height, xc_2, yc_2, conversion,ymax, x_right, y_right ):
    
    x_list=[]
    y_list=[]
    
    for i in range(ymax+1):
        x_list.append((points_x[i]))
        y_list.append((points_y[i]-points_y[ymax])*-1)
    
    
    
    x_right_new=[]
    x_min=min(x_right)
    for i in range(len(x_right)):
        x_right_new.append(x_right[i]-x_min)
    
    dist_im=[]
    for i in range(len(y_right)):
        dist_im.append(math.sqrt((y_right[i]**2)+(x_right_new[i]**2)))
    
    dist_ylp=[]
    for i in range(len(y_list)):
        dist_ylp.append(math.sqrt((y_list[i]**2)+(x_list[i]**2)))
        
    section=143
    
    ylp=chunkIt(dist_ylp, section)
    im=chunkIt(dist_im, section)
        
    
    #print((ylp))
    #print((im))
        
    chi,p=chisquare(f_obs=im,f_exp=ylp)
    print(chi, "chi3",p)  
    print(chi/(section),"chi4")
    
    plt.plot(x_list, y_list)
    plt.plot(x_right_new, y_right)
    #plt.title("droplet abc ")
    plt.show()
    
    
        
    return(chi)

def chi_fit3(points_x, points_y,circle_height, xc_2, yc_2, conversion,ymax, x_right, y_right, x_left, y_left):
    
    x_list=[]
    y_list=[]
    
    for i in range(ymax+1):
        x_list.append((points_x[i]))
        y_list.append((points_y[i]-points_y[ymax])*-1)
    
    x_left_new=[]
    xl_max=max(x_left)
    for i in range(len(x_left)):
        x_left_new.append((x_left[i]-xl_max)*-1)
    
    x_right_new=[]
    x_min=min(x_right)
    for i in range(len(x_right)):
        x_right_new.append(x_right[i]-x_min)
    
    dist_im=[]
    for i in range(len(y_right)):
        dist_im.append(math.sqrt((y_right[i]**2)+(x_right_new[i]**2)))
    
    dist_ylp=[]
    for i in range(len(y_list)):
        dist_ylp.append(math.sqrt((y_list[i]**2)+(x_list[i]**2)))
        
    section=180
    
    ylp=chunkIt(dist_ylp, section)
    im=chunkIt(dist_im, section)
        
    
    #print((ylp))
    #print((im))
        
    chi,p=chisquare(f_obs=im,f_exp=ylp)
    print(chi, "chi3",p)  
    print(chi/(section),"chi4")
    
    plt.plot(x_list, y_list)
    plt.plot(x_right_new, y_right)
    plt.plot(x_left_new, y_left)
    #plt.title("droplet abc ")
    plt.show()
    
    dist_im2=[]
    for i in range(len(y_left)):
        dist_im2.append(math.sqrt((y_left[i]**2)+(x_left_new[i]**2)))
    
    print(len(dist_im2))
        
    section2=180
    
    ylp=chunkIt(dist_ylp, section2)
    im2=chunkIt(dist_im2, section2)
        
    
    #print((ylp))
    #print((im))
        
    chi2,p2=chisquare(f_obs=im2,f_exp=ylp)
    print(chi2, "chi4",p2)  
    print(chi2/(section2),"chi5") 
    
    fobs=[]
    for i in range (len(im2)):
        fobs.append((im[i]+im2[i])/2)
    
    chi3,p3=chisquare(f_obs=fobs,f_exp=ylp)
    print(chi3, "chi6",p3)  
    print(chi3/(section2),"chi7") 
        
    return(chi)
       
def chunkIt(seq,num):
    avg = len(seq) / float(num)
    out = []
    
    last = 0.0

    while last < len(seq):
        a=seq[int(last):int(last + avg)]
        b=sum(a)/len(a)
        out.append(b)
        
        last += avg
        
    """
    new=[] 
    
    for i in range(out):
        new.append(sum(out[i])/len(out[i]))
    """
    #print(out,"ij")
    return out

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
plt.title("droplet data points")
plt.show()
#print(points_x)

#find values for the base width and the droplets height
height_droplet, base_width, width_droplet, points_x_left, points_x_right, points_y_left, points_y_right = droplet_values_alt(points_x, points_y) 
print(height_droplet, "droplet height")

print(len(points_x_right),"len")
#fit circle to the droplet using only the 10 values around the top point of the circle
x = r_[points_x[-10:]]
y = r_[points_y[-10:]]
circle_height, xc_2, yc_2 = calculate_circle_fit(points_x, points_y, x, y)

print(xc_2, yc_2, "centre")

circle1 = plt.Circle((xc_2, yc_2), circle_height, color='b', fill=False)
ax = plt.gca()
ax.cla() 
ax.set_xlim((0, 120))
ax.set_ylim((0, 150))
ax.scatter(points_x, points_y)
ax.add_artist(circle1)
plt.title("droplet with circle fit")
plt.show()

angle,error,var,x_list,y_list ,list_max= young_laplace_integration(circle_height, height_droplet, delta_angle, surface_tension, density, gravity, conversion, xc_2,yc_2)

plt.scatter(x_list, y_list)
#ax.add_artist(circle1)
plt.title("droplet ")
plt.show()

#chi=chi_fit(x_list, y_list,circle_height, xc_2, yc_2, conversion,list_max, points_x_right, points_y_right)
#chi=chi_fit2(x_list, y_list,circle_height, xc_2, yc_2, conversion,list_max, points_x_right, points_y_right)
hi=chi_fit3(x_list, y_list,circle_height, xc_2, yc_2, conversion,list_max, points_x_right, points_y_right, points_x_left, points_y_left)

print("angle;", angle)
print("standard deviation:", error)
print("Variance", var)
stop = timeit.default_timer()
print('Run time: ', stop - start) 
