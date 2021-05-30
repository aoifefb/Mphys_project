# Mphys_project
Code for finding contact angle of Sessile droplets on a flat surface. Best method is Young-Laplace

README file for Aoife baskill Master Project - Image Anyalysis for a Mobile Lab

Best method for calulating is young-Laplace method
Circle works nearly as good while tangnet method is bad

 For Young-Laplace method used inthegration along edge based on paper by F. Behroozi DOI:10.1119/1.5078512
 Calclating circle fit is based on method from https://github.com/scipy/scipy-cookbook/blob/master/ipython/Least_Squares_Circle.ipynb.
 It comes from the old scipy wiki which has been archived and the circles least squares circle method
 
 Mehtods are named with young-laplace (yl) for one image, young-lpalace glob (ylg) for method 
 using multiple images. Cicel method is named the same with circle method for one image (cm) and
 circle method glob (cmg) for multiple images in one folder
 
what to change in code for running it:

ylp and ylpg both contain constants density, gravity, surface tensiona and conversion from
metres to pixels. these are currently from the F.Behroozi paper. found in lines 21-25 

select file directory and name of file to write data to for ylpg lines 27, 28 and cmg lines 19,20

select pixel threshold to reduce noise in image cm line 44, cmg line 32, ylp line 38 and ylpg line 40

selct box area of intrest which droplet exists in  form (x1,y1,x2,y2) with 0,0 in top left corner of 
image and coordinated as top left to bottom right for the box. cm line 55, cmg line 43, ylp line 49 and ylpg line 51

select file to run for cm line 217 and ylp line 431. Allows most image file types (tested with Jpeg, Tiff and png) does not work with bmp type

for ylpg line 163 and ylp line 215 the method only loops 10000 time to intgrate down the edge of the droplet, this may need to 
be changed if the image is large or the droplet is large.

for sleecting the file type for the directory this is currently set to tiff and can be changed
in ylpg line 208 and cmg line 187

currently the graph outputs for the droplet have edges set but this may need to be changed if
the image is diffrent from initall set.

the chi-fit method used in ylp is chifit3 and is the only one that actually runs.
