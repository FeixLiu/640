import cv2
import os
import time


def splitFrames_mp4(sourceFileName, im_file):
    temp = []
    face_cascade = cv2.CascadeClassifier("./haarshare/haarcascade_frontalface_alt2.xml")
    video_path = os.path.join(im_file, sourceFileName + '.mp4')
    times = 0

    outPutDirName = im_file + '/video/' + sourceFileName + '/'

    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)

    camera = cv2.VideoCapture(video_path)
    while True:
        res, image = camera.read()
        if not res:
            break
        faces = face_cascade.detectMultiScale(image, 1.1, 5)
        if len(faces):
            for (x, y, w, h) in faces:
                times += 1
                if w >= 128 and h >= 128:
                    X = int(x)
                    W = min(int(x + w), image.shape[1])
                    Y = int(y)
                    H = min(int(y + h), image.shape[0])
                    f = cv2.resize(image[Y:H, X:W], (W - X, H - Y))
                    resized = cv2.resize(f, (256, 256))
                    temp.append(f)
                    cv2.imwrite(outPutDirName + str(times) + '.jpg', resized)
    camera.release()
    return temp


if __name__ == '__main__':
    """labels = {}
    csvFile = open("./640/Labels.csv", "r")
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        labels[item[0]] = item[1]
    csvFile.close()"""

    start = time.time()
    im_file = './640/presidential_videos/'
    im_file_done = './640/presidential_videos/video'
    i = 0
    done = os.listdir(im_file_done)
    for im_name in os.listdir(im_file_done):
        a = os.path.join(im_file_done, im_name)
        if len(os.listdir(a)) > 100:
            i += 1
        suffix_file = os.path.splitext(im_name)[-1]
        sourceFileName = os.path.splitext(im_name)[0]
        #if sourceFileName not in done:
            #faces = splitFrames_mp4(sourceFileName, im_file)
    end = time.time()
    print(i)
    print(end - start)

