import cv2
import os


def splitFrames_mp4(sourceFileName, im_file):
    video_path = os.path.join(im_file, sourceFileName + '.mp4')
    times = 0

    frameFrequency = 25
    outPutDirName = im_file + '/video/' + sourceFileName + '/'

    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)

    camera = cv2.VideoCapture(video_path)
    while True:
        times += 1
        res, image = camera.read()
        if not res:
            break
        if times % frameFrequency == 0:
            cv2.imwrite(outPutDirName + str(times)+'.jpg', image)
        #cv2.imwrite(outPutDirName + str(times) + '.jpg', image)
    camera.release()


if __name__ == '__main__':
    im_file = './640/presidential_video/'

    # for im_name in im_names:
    for im_name in os.listdir(im_file):
        suffix_file = os.path.splitext(im_name)[-1]
        sourceFileName = os.path.splitext(im_name)[0]
        splitFrames_mp4(sourceFileName, im_file)
