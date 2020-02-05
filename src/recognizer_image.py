import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')
#sys.path.append('../mtcnn')
sys.path.append('../retinaface')

from keras.models import load_model
#from mtcnn.mtcnn import MTCNN
from retinaface import RetinaFace
from imutils import paths
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import time
import dlib
import cv2
import os
from sys import getsizeof

ap = argparse.ArgumentParser()

ap.add_argument("--mymodel", default="outputs/my_model.h5",
    help="Path to recognizer model")
ap.add_argument("--le", default="outputs/le.pickle",
    help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
    help='Path to embeddings')
ap.add_argument("--image-out", default="outputs/001_out.jpg",
    help='Path to output video')
ap.add_argument("--image-in", default="../datasets/test/001.jpg")


ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Load embeddings and labels
data = pickle.loads(open(args.embeddings, "rb").read())
le = pickle.loads(open(args.le, "rb").read())

embeddings = np.array(data['embeddings'])
labels = le.fit_transform(data['names'])

# Initialize detector
#detector = MTCNN()

#retinaface detector initialization
thresh = 0.8
#scales = [720, 1280]
# scale for image resizing
scales = [1.0]
flip = False
gpuid = 0
detector = RetinaFace('../retinaface/pretrained_model/R50', 0, gpuid, 'net3')

# Initialize faces embedding model
print("--------args start---------")
print(args)
print("--------args end-----------")
embedding_model =face_model.FaceModel(args)

# Load the classifier model
model = load_model('outputs/my_model.h5')

# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)

# Initialize some useful arguments
cosine_threshold = 0.8
proba_threshold = 0.85
comparing_num = 5
frames = 0
#print(args.image_in)
img = cv2.imread(args.image_in)

#save_width = 1280
#save_height = 720
save_width = 112
save_height = 112

frame = img
print(frame.shape)
frame = cv2.resize(frame, (save_width, save_height))
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print('starting detection')
bboxes, landmarkses = detector.detect(frame, thresh, scales=scales, do_flip=flip)
print('ending detection')
counter = 0
for bbox in bboxes:  
  bbox = bbox.astype(int)
  landmarks = landmarkses[counter]
  counter + 1
  landmarks = np.array([landmarks[0][0], landmarks[1][0], landmarks[2][0], landmarks[3][0], landmarks[4][0],
                      landmarks[0][1], landmarks[1][1], landmarks[2][1], landmarks[3][1], landmarks[4][1]])
  landmarks = landmarks.reshape((2,5)).T
  nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
  nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
  nimg = np.transpose(nimg, (2,0,1))
  #print(nimg.shape)
  embedding = embedding_model.get_feature(nimg)
  #print(embedding.shape)
  embedding = embedding_model.get_feature(nimg).reshape(1,-1)
  #print(embedding.shape)
  text = "Unknown"
  # Predict class
  #print(embedding.shape)
  preds = model.predict(embedding)
  preds = preds.flatten()
  # Get the highest accuracy embedded vector
  j = np.argmax(preds)
  proba = preds[j]
  # Compare this vector to source class vectors to verify it is actual belong to this class
  match_class_idx = (labels == j)
  match_class_idx = np.where(match_class_idx)[0]
  selected_idx = np.random.choice(match_class_idx, comparing_num)
  compare_embeddings = embeddings[selected_idx]
  # Calculate cosine similarity
  cos_similarity = CosineSimilarity(embedding, compare_embeddings)
  if cos_similarity < cosine_threshold and proba > proba_threshold:
      name = le.classes_[j]
      text = "{}".format(name)
      print("Recognized: {} <{:.2f}>".format(name, proba*100))
  # Start tracking
  #texts.append(text)
  y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
  cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
  cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)

cv2.imwrite(args.image_out, frame)
