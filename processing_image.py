
#THIRD PARTY LIBRARIES
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

#STANDARD LIBRARIES
from math import floor
import sys

#PREPROCESS DATA FOR UNSUPERVISED CLASSIFICATION
im = Image.open(sys.argv[1])
im_width = im.size[0]
im_height = im.size[1]
pixels = im.load()
clustering_data = []

for x in range(im_width):
	for y in range(im_height):
		#ADD PIXEL DATA TO 
		clustering_data.append([pixels[x,y][0],pixels[x,y][1],pixels[x,y][2]])

#CLUSTER PIXEL DATA USING K-MEANS
kmeans = KMeans(n_clusters = int(sys.argv[2])).fit(clustering_data)
kmeans_color_values = kmeans.cluster_centers_

#CREATE RESULTING IMAGE FOR K-MEANS CLUSTERING
image_representation = np.zeros([im_height, im_width, 3], dtype=np.uint8)
for x in range(im_width):
	for y in range(im_height):
		pixel_data = [pixels[x,y][0], pixels[x,y][1], pixels[x,y][2]]
		cluster = kmeans.predict([pixel_data])
		kmeans_cluster_values = kmeans_color_values[cluster[0]]
		image_representation[y,x] = [
			floor(kmeans_cluster_values[0]), 
			floor(kmeans_cluster_values[1]), 
			floor(kmeans_cluster_values[2])
		]

kmeans_image = Image.fromarray(image_representation,"RGB")
kmeans_image.save(sys.argv[3])
