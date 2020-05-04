import cv2 as cv
import numpy as np
import time

img_gray = cv.imread("im47_t.bmp", cv.IMREAD_GRAYSCALE)
img_rgb = cv.imread("im47_t.bmp", cv.IMREAD_COLOR)

resize_percentage = 100
max_dist = 150

global matched_pts
matched_pts = []
##--------------------


img_gray = cv.resize(img_gray, (
    int(img_gray.shape[1] * resize_percentage / 100),
    int(img_gray.shape[0] * resize_percentage / 100)))

img_rgb = cv.resize(img_rgb, (
    int(img_rgb.shape[1] * resize_percentage / 100), int(img_rgb.shape[0] * resize_percentage / 100)))

sift = cv.xfeatures2d.SIFT_create()
keypoints_sift, descriptors = sift.detectAndCompute(img_gray, None)
img = cv.drawKeypoints(img_rgb, keypoints_sift, None)
cv.imshow(" All Keypoints ", img)

pts1 = np.array([], np.int32)
pts2 = np.array([], np.int32)


def compare_keypoint(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2)


def apply_sift():
    for index_dis, key_desc_dis in enumerate(descriptors):  # dist = numpy.linalg.norm(a-b)
        for index_ic in range(index_dis + 1, len(keypoints_sift)):
            point1_x = int(round(keypoints_sift[index_dis].pt[0]))
            point1_y = int(round(keypoints_sift[index_dis].pt[1]))
            point2_x = int(round(keypoints_sift[index_ic].pt[0]))
            point2_y = int(round(keypoints_sift[index_ic].pt[1]))
            if point1_x == point2_x & point1_y == point2_y:
                # print("benzer keypoints")
                continue

            dist = compare_keypoint(key_desc_dis, descriptors[index_ic])

            if dist < max_dist:
                matched_pts.append([round(keypoints_sift[index_dis].pt[0]), round(keypoints_sift[index_dis].pt[1]),
                                    round(keypoints_sift[index_ic].pt[0]), round(keypoints_sift[index_ic].pt[1])])


def draw(matched_pts):
    for points in matched_pts:
        cv.circle(img_rgb, (points[0], points[1]),
                  4,
                  (0, 0, 255),
                  -1)  # eslesen objeyi isaretlemek icin

        cv.circle(img_rgb, (points[2], points[3]),
                  4,
                  (255, 0, 0),
                  -1)  # eslesen objeyi isaretlemek icin

        img_line = cv.line(img_rgb,
                           (points[0], points[1]),
                           (points[2], points[3]),
                           (0, 255, 0), 1)
    cv.imshow("Image", img_rgb)


if __name__ == '__main__':
    time1 = time.time()

    apply_sift()

    time2 = time.time()
    print('{:.3f} ms'.format((time2 - time1) * 1000.0))

    draw(matched_pts)

    cv.waitKey(0)
    cv.destroyAllWindows()
