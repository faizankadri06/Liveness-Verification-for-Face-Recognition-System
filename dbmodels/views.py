from django.shortcuts import render, redirect, HttpResponse
from django.template import loader
from .models import Login,ALogin,User
import os
import face_recognition
import tensorflow as tf
import numpy as np
import imutils
import pickle
import time
import cv2
from django.http import StreamingHttpResponse

def user_login(request):
    return render(request, 'user_login.html')


def signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        try:
            user=Login.loginauth_objects.get(username=username,password=password)
            if user is not None:
                query_res = User.objects.filter(username=username).values()
                template = loader.get_template('Dashboard.html')
                context = {
                    'mymembers': query_res,
                }
                return HttpResponse(template.render(context, request))
                #return render(request,'Dashboard.html',{})
            else:
                return redirect('/')
        except Exception as Identifier:
            return redirect('/')

    else:
        return render(request, 'user_login.html')
  

def admin(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        try:
            user=ALogin.loginauth_objects.get(username=username,password=password)
            if user is not None:
                query_res = User.objects.all()
                return render(request,'admin_dash.html',{'query_res':query_res})
            else:
                print("Incorrect Combination of username and password")

                return redirect('/')
        except Exception as Identifier:
            return redirect('/')

    else:
        return render(request, 'user_login.html')


def index(request):
    return render(request,'index.html')

def Live(request):
    vs = cv2.VideoCapture(0)

    def webcam_stream():
        args = {'model':'dbmodels/liveness.model' , 'le':'dbmodels/label_encoder.pickle' , 'detector':'dbmodels/face_detector',
                'encodings':'../face_recognition/encoded_faces.pickle', 'confidence':0.5}

        # load the encoded faces and names
        print('[INFO] loading encodings...')
        with open('dbmodels/encoded_faces.pickle', 'rb') as file:
            encoded_data = pickle.loads(file.read())
        # load our serialized face detector from disk
        print('[INFO] loading face detector...')
        proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
        model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
        detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

        # load the liveness detector model and label encoder from disk
        liveness_model = tf.keras.models.load_model(args['model'])
        le = pickle.loads(open(args['le'], 'rb').read())

        # initialize the video stream and allow camera to warmup
        print('[INFO] starting video stream...')

        time.sleep(2) # wait camera to warmup
        # count the sequence that person appears
        # this is just to make sure of that person and to show how model works
        # you can delete this if you want
        sequence_count = 0

        # initialize variables needed to return
        # in case, users press 'q' before the program process the frame
        name = 'Unknown'
        label_name = 'fake'

        # iterate over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream
            # and resize it to have a maximum width of 600 pixels
            ret, frame = vs.read()
            frame = imutils.resize(frame, width=800)
            cv2.putText(frame, "Press 'q' to quit", (20,35), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0), 2)
            # grab the frame dimensions and convert it to a blob
            # blob is used to preprocess image to be easy to read for NN
            # basically, it does mean subtraction and scaling
            # (104.0, 177.0, 123.0) is the mean of image in FaceNet
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network
            # and obtain the detections and predictions
            detector_net.setInput(blob)
            detections = detector_net.forward()

            # iterate over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e. probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > args['confidence']:
                    # compute the (x,y) coordinates of the bounding box
                    # for the face and extract the face ROI
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')

                    # expand the bounding box a bit
                    # (from experiment, the model works better this way)
                    # and ensure that the bounding box does not fall outside of the frame
                    startX = max(0, startX-20)
                    startY = max(0, startY-20)
                    endX = min(w, endX+20)
                    endY = min(h, endY+20)

                    # extract the face ROI and then preprocess it
                    # in the same manner as our training data
                    face = frame[startY:endY, startX:endX] # for liveness detection
                    # expand the bounding box so that the model can recog easier
                    face_to_recog = face # for recognition
                    # some error occur here if my face is out of frame and comeback in the frame
                    try:
                        face = cv2.resize(face, (32,32)) # our liveness model expect 32x32 input
                    except:
                        break

                    # face recognition
                    rgb = cv2.cvtColor(face_to_recog, cv2.COLOR_BGR2RGB)
                    #rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb)
                    # initialize the default name if it doesn't found a face for detected faces
                    name = 'Unknown'
                    # loop over the encoded faces (even it's only 1 face in one bounding box)
                    # this is just a convention for other works with this kind of model
                    for encoding in encodings:
                        matches = face_recognition.compare_faces(encoded_data['encodings'], encoding)

                        # check whether we found a matched face
                        if True in matches:
                            # find the indexes of all matched faces then initialize a dict
                            # to count the total number of times each face was matched
                            matchedIdxs = [i for i, b in enumerate(matches) if b]
                            counts = {}

                            # loop over matched indexes and count
                            for i in matchedIdxs:
                                name = encoded_data['names'][i]
                                counts[name] = counts.get(name, 0) + 1

                            # get the name with the most count
                            name = max(counts, key=counts.get)

                    face = face.astype('float') / 255.0
                    face = tf.keras.preprocessing.image.img_to_array(face)
                    # tf model require batch of data to feed in
                    # so if we need only one image at a time, we have to add one more dimension
                    # in this case it's the same with [face]
                    face = np.expand_dims(face, axis=0)

                    # pass the face ROI through the trained liveness detection model
                    # to determine if the face is 'real' or 'fake'
                    # predict return 2 value for each example (because in the model we have 2 output classes)
                    # the first value stores the prob of being real, the second value stores the prob of being fake
                    # so argmax will pick the one with highest prob
                    # we care only first output (since we have only 1 input)
                    preds = liveness_model.predict(face)[0]
                    j = np.argmax(preds)
                    label_name = le.classes_[j] # get label of predicted class

                    # draw the label and bounding box on the frame
                    label = f'{label_name}: {preds[j]:.4f}'
                    if name == 'Unknown' or label_name == 'fake':
                        sequence_count = 0
                    else:
                        sequence_count += 1
                    print(f'[INFO] {name}, {label_name}, seq: {sequence_count}')

                    if label_name == 'fake':
                        cv2.putText(frame, "Don't try to Spoof !", (startX, endY + 25),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)

                    cv2.putText(frame, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)

            # show the output fame and wait for a key press


            #cv2.imshow('Frame', frame)
            #key = cv2.waitKey(1) & 0xFF

            ret, buffer = cv2.imencode('.jpg', frame)

            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # if 'f' is pressed, stop the loop
            # if that person appears 10 frames in a row, stop the loop
            # you can change this if your GPU run faster

            if sequence_count==40:
                break

        # cleanup
        # vs.stop()
        # cv2.destroyAllWindows()
        vs.release()
    # return JsonResponse({'frame': frame})
        # # have some times for camera and CUDA to close normally
        # # (it can f*ck up GPU sometimes if you don't have high performance GPU like me LOL)
        # time.sleep(2)
    return StreamingHttpResponse(webcam_stream(), content_type='multipart/x-mixed-replace; boundary=frame')