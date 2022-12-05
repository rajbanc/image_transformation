import numpy as np
import cv2
import matplotlib.pyplot as plt

# read the input image
img = cv2.imread("lena.jpg")
# convert from BGR to RGB so we can plot using matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# disable x & y axis
plt.axis('on')
# show the image
plt.imshow(img)
plt.show()
# get the image shape
rows, cols, dim = img.shape
# transformation matrix for translation
M = np.float32([[1, 0, 60],
                [0, 1, 60],
                [0, 0, 1]])
#transformation matrix for scaling
M=np.float32([[1.5, 0 , 0],
              [0,  1.8, 0],
              [0,  0,   1]]) 
#shearing applied to x axis
M=np.float32([[1, 0.5, 0],
              [0, 1  , 0],
              [0, 0  , 1]]) 
#transformation matrix for x-axis reflection
M = np.float32([[1,  0, 0   ],
                [0, -1, rows],
                [0,  0, 1   ]])
                # transformation matrix for y-axis reflection
M = np.float32([[-1, 0, cols],
                [ 0, 1, 0   ],
                [ 0, 0, 1   ]])
                # change that if you will, just make sure you don't exceed cols & rows
cropped_img = img[0:400, 100:400]

#angle from degree to radian
angle = np.radians(10)

#transformation matrix for Rotation
""" M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
            	[np.sin(angle), np.cos(angle), 0],
            	[0, 0, 1]])
# apply a perspective transformation to the image
rotated_img = cv2.warpPerspective(img, M, (int(cols),int(rows))) """
# apply a perspective transformation to the image
reflected_img = cv2.warpPerspective(img,M,(int(rows),int(cols)))
                                         

# apply a perspective transformation to the image
translated_img = cv2.warpPerspective(img, M, (cols, rows))


#apply a perspective scaling to the image
scaled_img=cv2.warpPerspective(img,M,(cols*2,rows*2))


#apply a perspective transformation to the image
sheared_img=cv2.warpPerspective(img,M,(int(cols*1.5),int(rows*1.5)))


# apply a perspective transformation to the image
reflected_img = cv2.warpPerspective(img,M,(int(cols),int(rows)))

# disable x & y axis
plt.axis('on')

# show the resulting image
plt.imshow(translated_img)
plt.show()
plt.imshow(scaled_img)
plt.show()
plt.imshow(sheared_img)
plt.show()

# show the resulting image
plt.imshow(reflected_img)
plt.show()

# show the resulting image
plt.imshow(reflected_img)
plt.show()

# show the resulting image
plt.imshow(cropped_img)
plt.show()

# show the resulting image
""" plt.imshow(rotated_img)
plt.show() """

# save the resulting image to disk
plt.imsave("fruit_translated.jpg", translated_img)
plt.imsave("scaled_img.jpg",scaled_img)
plt.imsave("sheared_img.jpg",sheared_img)
plt.imsave("fruit_reflected.jpg", reflected_img)
plt.imsave("fruit_reflected.jpg", reflected_img)


# save the resulting image to disk
plt.imsave("fruits_cropped.jpg", cropped_img)

# save the resulting image to disk
#plt.imsave("city_rotated.jpg", rotated_img)