import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage.transform import resize
import math
from skimage.feature import hog
import numpy as np
import os
import cv2

#function to read_images from folder
def read_images_in_folder(folder_path):
    images = os.listdir(folder_path)     
    return images

folder_path = 'images'
water_folder_path = 'images-water'
images = read_images_in_folder(folder_path)
images_water = read_images_in_folder(water_folder_path)
print("total water images taken : ",len(images_water))
print("total images taken : ", len(images))#printing the length of actual water images and total_images

img=[]
img_water = []
for i in range(len(images)):
    img.append(resize(color.rgb2gray(io.imread(folder_path+'/'+images[i])), (128, 128)))
for j in range(len(images_water)):
    img_water.append(resize(color.rgb2gray(io.imread(water_folder_path+'/'+images_water[j])), (128, 128))) 
#getting images from the folder
img = np.array(img)
img_water = np.array(img_water)

def mag_and_angle_cal(mag,theta,imgs):
  for idx in range(imgs.shape[0]):
    for i in range(128):
      for j in range(128):
          # Condition for axis 0
        if j-1 <= 0 or j+1 >= 128:
          if j-1 <= 0:
            # Condition if first element
            Gx = imgs[idx][i][j+1] - 0
          elif j + 1 >= len(imgs[idx][0]):
            Gx = 0 - imgs[idx][i][j-1]
        # Condition for first element
        else:
          Gx = imgs[idx][i][j+1] - imgs[idx][i][j-1]
    
       # Condition for axis 1
        if i-1 <= 0 or i+1 >= 128:
          if i-1 <= 0:
            Gy = 0 - imgs[idx][i+1][j]
          elif i +1 >= 128:
            Gy =imgs[idx][i-1][j] - 0
        else:
          Gy = imgs[idx][i-1][j] - imgs[idx][i+1][j]

        # Calculating magnitude
        magnitude = math.sqrt(pow(Gx, 2) + pow(Gy, 2))
        mag[idx][i][j] += round(magnitude, 9)

        # Calculating angle
        if Gx == 0:
          angle = math.degrees(0.0)
        else:
          angle = math.degrees(abs(math.atan(Gy / Gx)))
        theta[idx][i][j] += round(angle, 9)

  return mag,theta   #returning the value of calculated angle and magnitude

mag = np.zeros(((len(images),128,128)))
theta = np.zeros(((len(images),128,128)))

mag_water = np.zeros(((len(images_water),128,128)))
theta_water = np.zeros(((len(images_water),128,128)))

mag_f,theta_f = mag_and_angle_cal(mag,theta,img)
mag_water_f,theta_water_f = mag_and_angle_cal(mag_water,theta_water,img_water)

#calling the function created above to get the magnitude and theta for total and water images

plt.figure(figsize=(15, 8))
plt.imshow(mag_f[1], cmap="gray")
plt.axis("off")
plt.show()  #just showing how the grayscale image looks for magnitude

plt.figure(figsize=(15, 8))
plt.imshow(theta[1], cmap="gray")
plt.axis("off")
plt.show() #just showing how the grayscale image looks for theta

number_of_bins = 9
step_size = 180 / number_of_bins #calculating the number of bins

def calculate_j(angle):
  temp = (angle / step_size) - 0.5
  j = math.floor(temp)
  return j

def calculate_Cj(j):
  Cj = step_size * (j + 0.5)
  return round(Cj, 9)

def calculate_value_j(magnitude, angle, j):
  Cj = calculate_Cj(j+1)
  Vj = magnitude * ((Cj - angle) / step_size)
  return round(Vj, 9)


def histogram(mag, theta, number_of_bins=9):
    hist_set = {}
    for idx in range(mag.shape[0]):
        histogram_points_nine = []
        for i in range(0, 128, 8):
            temp = []
            for j in range(0, 128, 8):
                magnitude_values = mag[idx, i:i+8, j:j+8]
                angle_values = theta[idx, i:i+8, j:j+8]
                bins = [0.0 for _ in range(number_of_bins)]
                for k in range(len(magnitude_values)):
                    for l in range(len(magnitude_values[0])):
                        value_j = calculate_j(angle_values[k, l])
                        Vj = calculate_value_j(magnitude_values[k, l], angle_values[k, l], value_j)
                        Vj_1 = magnitude_values[k, l] - Vj
                        bins[value_j] += Vj
                        bins[value_j + 1] += Vj_1
                bins = [round(x, 9) for x in bins]
                temp.append(bins)
            histogram_points_nine.append(temp)
        hist_set[idx] = histogram_points_nine
    return hist_set
#calulating the histograms for each 8X8 block of an image and then returning the histogram set for all images

hist = histogram(mag_f,theta_f)
hist_water = histogram(mag_water_f,theta_water_f)

epsilon = 1e-05
     
def features(hist):
  feature_set ={}
  for idx in range(len(hist)):
    feature_vectors = []
    for i in range(0, len(hist[0]) - 1, 1):
      temp = []
      for j in range(0, len(hist[0][0]) - 1, 1):
        values = [[hist[idx][i][x] for x in range(j, j+2)] for i in range(i, i+2)]
        final_vector = []
        for k in values:
          for l in k:
            for m in l:
              final_vector.append(m)
        k = round(math.sqrt(sum([pow(x, 2) for x in final_vector])), 9)
        final_vector = [round(x/(k + epsilon), 9) for x in final_vector]
        temp.append(final_vector)
      feature_vectors.append(temp)
    feature_set[idx] = feature_vectors
  return feature_set
#function for calculating the feature vector set for each image using the histogram obtained for each block 
#feature vector is an array of size = 8100 as we are resizing the image to 128x128

feature_vecs = features(hist)
feature_vecs_water= features(hist_water) #getting feature vectors for water images and total images

print(f'Number of HOG features = {len(feature_vecs[0]) * len(feature_vecs[0][0]) * len(feature_vecs[0][0][0])}')
print(f'Number of HOG features in water images = {len(feature_vecs_water[0]) * len(feature_vecs_water[0][0]) * len(feature_vecs_water[0][0][0])}')

Hog_features = len(feature_vecs[0]) * len(feature_vecs[0][0]) * len(feature_vecs[0][0][0])
Hog_features_water= len(feature_vecs_water[0]) * len(feature_vecs_water[0][0]) * len(feature_vecs_water[0][0][0])
#both are 8100 in our case

# flattended_fv=[]
for i in range(len(feature_vecs)):
    feature_vecs[i] = np.array(feature_vecs[i]).ravel()
for j in range(len(images_water)):
    feature_vecs_water[j] = np.array(feature_vecs_water[j]).ravel()
#flattening the featur_vecs array


feature_vecs_arr = np.zeros(( len(images), Hog_features))
feature_vecs_water_arr = np.zeros(( len(images_water), Hog_features_water ))

for i in range(len(feature_vecs)):
    for j in range(len(feature_vecs[0])):
        feature_vecs_arr[i][j]  += feature_vecs[i][j]
for i in range(len(feature_vecs_water)):
    for j in range(len(feature_vecs_water[0])):
        feature_vecs_water_arr[i][j] += feature_vecs_water[i][j]
#forming 2d array of feature vectors for actual water images and total images


y_minus = np.zeros(len(images)) - 1
y_plus = np.zeros(len(images_water)) + 1
y_minus = y_minus.T
y_minus = y_minus.reshape((len(images),1))
labelled_1 = np.hstack((feature_vecs_arr,y_minus))
y_plus = y_plus.reshape((len(images_water),1))
labelled_2 = np.hstack((feature_vecs_water_arr,y_plus))
labelled = np.vstack((labelled_1,labelled_2))
np.random.shuffle(labelled)
#labelling the actual water images as 1 and rest as -1 and then making a shuffled input data 
np.save("data.npy",labelled) #data pre processing done
