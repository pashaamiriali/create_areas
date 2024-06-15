# create areas based on colors on images

### to use this code you will first need an image file that has parts of it highlighted or it's pre-colored in some ares. then you will need to find out what is that specific color (I use colorzilla). convert it into RGB and then put it in lower and upper colors. 

### subtract 1 from the original number and put it in the lower bounds and add 1 to the color and put it in upper color. then run the code and expect a result in json. the result contains a list of polygons that are determined by a list of dx and dy. You can then use the points to draw the polygons in you flutter app.