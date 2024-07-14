import cv2 
import numpy as np 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage.transform import resize
import math
from skimage.feature import hog
import numpy as np
  

def calculate_feature_vector(x, y, blk_size, cropped_image):
    img = resize(color.rgb2gray(io.imread(cropped_image)), (x,y))

    img = np.array(img)

    mag = []
    theta = []
    for i in range(x):
        magnitudeArray = []
        angleArray = []
        for j in range(y):
        # Condition for axis 0
            if j-1 <= 0 or j+1 >= y:
                if j-1 <= 0:
            # Condition if first element
                    Gx = img[i][j+1] - 0
                elif j + 1 >= len(img[0]):
                    Gx = 0 - img[i][j-1]
        # Condition for first element
                else:
                    Gx = img[i][j+1] - img[i][j-1]
        
        # Condition for axis 1
            if i-1 <= 0 or i+1 >= x:
                if i-1 <= 0:
                    Gy = 0 - img[i+1][j]
                elif i +1 >= x:
                    Gy = img[i-1][j] - 0
                else:
                    Gy = img[i-1][j] - img[i+1][j]

        # Calculating magnitude
            magnitude = math.sqrt(pow(Gx, 2) + pow(Gy, 2))
            magnitudeArray.append(round(magnitude, 9))

        # Calculating angle
            if Gx == 0:
                angle = math.degrees(0.0)
            else:
                angle = math.degrees(abs(math.atan(Gy / Gx)))
            angleArray.append(round(angle, 9))
        mag.append(magnitudeArray)
        theta.append(angleArray)
        

    mag = np.array(mag)
    theta = np.array(theta)

    number_of_bins = 9
    step_size = 180 / number_of_bins

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
        

    histogram_points_nine = []
    for i in range(0, x, blk_size):
        temp = []
        for j in range(0, y, blk_size ):
            magnitude_values = [[mag[i][x] for x in range(j, j+8)] for i in range(i,i+8)]
            angle_values = [[theta[i][x] for x in range(j, j+8)] for i in range(i, i+8)]
            for k in range(len(magnitude_values)):
                for l in range(len(magnitude_values[0])):
                    bins = [0.0 for _ in range(number_of_bins)]
                    value_j = calculate_j(angle_values[k][l])
                    Vj = calculate_value_j(magnitude_values[k][l], angle_values[k][l], value_j)
                    Vj_1 = magnitude_values[k][l] - Vj
                    bins[value_j]+=Vj
                    bins[value_j+1]+=Vj_1
                    bins = [round(x, 9) for x in bins]
            temp.append(bins)
        histogram_points_nine.append(temp)

    epsilon = 1e-05
        

    feature_vectors = []
    for i in range(0, len(histogram_points_nine) - 1, 1):
        temp = []
        for j in range(0, len(histogram_points_nine[0]) - 1, 1):
            values = [[histogram_points_nine[i][x] for x in range(j, j+2)] for i in range(i, i+2)]
            final_vector = []
            for k in values:
                for l in k:
                    for m in l:
                        final_vector.append(m)
            k = round(math.sqrt(sum([pow(x, 2) for x in final_vector])), 9)
            final_vector = [round(x/(k + epsilon), 9) for x in final_vector]
            temp.append(final_vector)
        feature_vectors.append(temp)
    fv=np.array(feature_vectors).ravel()
    sp=fv.shape
    fv=fv.reshape((1,sp[0]))
    return fv  #function for calculating the feature vector for input images

#here we are getting the cropped image using mouse selection


cropped_image = 'crop.jpg'
feature_vector_arr=calculate_feature_vector(128,128,8, cropped_image)
#getting the feature vector array


# Load the labelled data
data = np.load("data.npy")

# Split the data into features (X) and labels (y)
X = data[:, :-1]
y = data[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')

# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = svm_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100, "%")

#here we have trained our svm using the hog featured vector 
y_pred_test = svm_classifier.predict(feature_vector_arr)  #now we are testing over our test image
# print(y_pred_test)
if(y_pred_test< 0):
    print("The image is not of a water body")
else:
    print("The image is of a water body")
