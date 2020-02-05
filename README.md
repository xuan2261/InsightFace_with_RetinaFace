# Face Recognition with InsightFace
Recognize and manipulate faces with Python and its support libraries.  
The project uses [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace) for detecting faces, then applies a simple alignment for each detected face and feeds those aligned faces into embeddings model provided by [InsightFace](https://github.com/deepinsight/insightface). Finally, a softmax classifier was put on top of embedded vectors for classification task.

## Getting started
### Requirements
- Python 3.3+
- Sklearn 0.20.1
- Virtualenv
- python-pip
- mx-net
- tensorflow
- macOS or Linux
## Usage
First, go to directory that you have cloned, activate __env__ to use installed package, alternatively, you must install all 
```
python3 recognizer_image.py 
```
This will perform recognition on the image and save an output image in the `InsightFace_with_RetinaFace/src/output`

You can also try with recognition in video with:
```
python3 recognizer_video.py
```
This will perform recognition
or streaming if your machine supports camera:
```
python3 recognizer_stream.py
```
A Google Colab Notebook is also provided that has a complete set of steps to perform recognition on both images and videos.
The Notebook's name is `InsightFace_with_RetinaFace.ipynb`

## Build your own faces recognition system
By default, most of the input and output arguments were provided, models and embeddings is set default stored in `/src/outputs/`.  
### 1. Prepare your data 
Our training datasets were built as following structure:
```
/datasets
  /train
    /person1
      + face_01.jpg
      + face_02.jpg
      + ...
    /person2
      + face_01.jpg
      + face_02.jpg
      + ...
    / ...
  /test
  /unlabeled_faces
  /videos_input
  /videos_output
```
In each `/person_x` folder, put your face images corresponding to _person_name_ that has been resized to _112x112_ (input size for InsightFace). Here I provided two ways to get faces data from your webcam and video stored in your storage.  
__a. Get faces from camera__  
Run following command, with `--faces` defines how many faces you want to get, _default_ is 20
```
python3 get_faces_from_camera.py [--faces 'num_faces'] [--output 'path/to/output/folder']
```
Here `[--cmd]` means _cmd_ is optional, if not provide, script will run with its default settings.  
__b. Get faces from video__  
Prepare a video that contains face of the person you want to get and give the path to it to `--video` argument:
```
python3 get_faces_from_video.py [--video 'path/to/input/video'] [--output 'path/to/output/folder']
``` 
As I don't provide stop condition to this script, so that you can get as many faces as you want, you can also press __q__ button to stop the process.</br>
  
The default output folder is `/unlabeled_faces`, select all faces that match the person you want, and copy them to `person_name` folder in `train`. Do the same things for others person to build your favorite datasets.
### 2. Generate face embeddings
```
python3 faces_embedding.py [--dataset 'path/to/train/dataset'] [--output 'path/to/out/put/model']
```
### 3. Train classifier with softmax
```
python3 train_softmax.py [--embeddings 'path/to/embeddings/file'] [--model 'path/to/output/classifier_model'] [--le 'path/to/output/label_encoder']
```

### 4. Run
Yep!! Now you have a trained model, let's enjoy it!  
Face recognization with image as input:
```
python3 recognizer_image.py [--image-in 'path/to/test/image'] [...]
```
Face recognization with video as input:
```
python3 recognizer_video.py [--video 'path/to/test/video'] [...]
```
Face recognization with camera:
```
python3 recognizer_stream.py
```
`[...]` means other arguments, I don't provide it here, you can look up in the script at arguments part
