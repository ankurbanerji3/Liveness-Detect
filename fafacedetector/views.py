from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.http import JsonResponse
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from imutils import paths
import face_recognition
import numpy as np
import argparse
import imutils
import time
import dlib
import urllib
import json
import cv2
import os

'''FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_alt.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

@csrf_exempt
def detect(request):
	data = {"success": False}
	if request.method == "POST":
		if request.FILES.get("image", None) is not None:
			image = _grab_image(stream=request.FILES["image"])
		else:
			url = request.POST.get("url", None)
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)
			image = _grab_image(url=url)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
		rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
		rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
		data.update({"num_faces": len(rects), "faces": rects, "success": True})

	return JsonResponse(data)

def _grab_image(path=None, stream=None, url=None):
	#url = "https://cdn.cnn.com/cnnnext/dam/assets/200126175745-barack-obama-102618-file-super-tease.jpg"
	if path is not None:
		image = cv2.imread(path)
	else:	
		if url is not None:
			with urllib.request.urlopen(url) as urll:
				data = urll.read()
			#resp = urllib.request.urlopen(url)
			#data = resp.read()
		elif stream is not None:
			data = stream.read()
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	return image'''

def collect():
	imagePaths = list(paths.list_images("{base_path}/cascades/5-celebrity-faces-dataset-new/data/train".format(base_path=os.path.abspath(os.path.dirname(__file__)))))
	#mmod_human_face_detector.dat
	knownEncodings = []
	knownNames = []
	for (i, imagePath) in enumerate(imagePaths):
		name = imagePath.split(os.path.sep)[-2]
		image = cv2.imread(imagePath)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		boxes = face_recognition.face_locations(rgb, model="{base_path}/cascades/mmod_human_face_detector.dat".format(base_path=os.path.abspath(os.path.dirname(__file__))))
		encodings = face_recognition.face_encodings(rgb, boxes)
		for encoding in encodings:
			knownEncodings.append(encoding)
			knownNames.append(name)
	data = {"encodings": knownEncodings, "names": knownNames}
	return data

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

@csrf_exempt
def process_and_show(request):
	data = {"success": False, "name": "Unknown", "Blinks": 0}
	collect_data = collect()
	if request.method == "POST":
		EYE_AR_THRESH = 0.3
		EYE_AR_CONSEC_FRAMES = 3
		COUNTER = 0
		TOTAL = 0
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor("{base_path}/cascades/shape_predictor_68_face_landmarks.dat".format(base_path=os.path.abspath(os.path.dirname(__file__))))
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
		vs = cv2.VideoCapture("{base_path}/media/VID_20200511_131132773.mp4".format(base_path=os.path.abspath(os.path.dirname(__file__))))
		face_cascPath = "{base_path}/cascades/haarcascade_frontalface_alt.xml".format(base_path=os.path.abspath(os.path.dirname(__file__)))
		face_detector = cv2.CascadeClassifier(face_cascPath)
		while True:
			ret, frame = vs.read()
			if not ret:
				break
			frame = imutils.resize(frame, width=450)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			rects = detector(gray, 0)
			faces = face_detector.detectMultiScale(
				gray,
				scaleFactor = 1.2,
				minNeighbors = 5,
				minSize = (50, 50),
				flags = cv2.CASCADE_SCALE_IMAGE
			)
			for (i, rect) in enumerate(rects):
				(x,y,w,h) = faces[i]
				encoding = face_recognition.face_encodings(rgb, [(y, x+w, y+h, x)])[0]
				matches = face_recognition.compare_faces(collect_data["encodings"], encoding)
				name = "Unknown"
				if True in matches:
					matchedIdxs = [i for (i, b) in enumerate(matches) if b]
					counts = {}
					for j in matchedIdxs:
						name = collect_data["names"][j]
						counts[name] = counts.get(name, 0) + 1
					name = max(counts, key=counts.get)
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)
				ear = (leftEAR + rightEAR) / 2.0
				if ear < EYE_AR_THRESH:
					COUNTER += 1
				else:
					if COUNTER >= EYE_AR_CONSEC_FRAMES:
						TOTAL += 1
					COUNTER = 0
				if TOTAL >= 1:
					data["name"] = name
					data["Blinks"] = TOTAL
					data["success"] = True
	return data


class FileView(APIView):
	parser_classes = (MultiPartParser, FormParser)
	def post(self, request, *args, **kwargs):
		file_serializer = FileSerializer(data=request.data)
		if file_serializer.is_valid():
			file_serializer.save()
			data = process_and_show(request)
			return JsonResponse(data)
			#return Response(file_serializer.data, status=status.HTTP_201_CREATED)
		else:
			return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
