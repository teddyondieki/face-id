from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
from matplotlib import pyplot
from os import listdir
import cv2
from os.path import isdir
from numpy import savez_compressed

TRAIN_DIR = 'data/att_faces/train'
VAL_DIR = 'data/att_faces/val'

def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def load_faces(directory):
    faces = list()

    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)

    return faces


def load_dataset(directory):
    x, y = list(), list()

    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('loaded {} examples for class: {}'.format(len(faces), subdir))
        x.extend(faces)
        y.extend(labels)
    return asarray(x), asarray(y)


trainX, trainY = load_dataset('/home/teddy/Desktop/Lima_Tech/face_id_tutorial/data/att_faces/train/')
print(trainX.shape, trainY.shape)
testX, testY = load_dataset('/home/teddy/Desktop/Lima_Tech/face_id_tutorial/data/att_faces/val/')
print(testX.shape, testY.shape)
savez_compressed('/home/teddy/Desktop/Lima_Tech/face_id_tutorial/data/faces-dataset.npz', trainX, trainY, testX, testY)
