import cv2
import typing
from matplotlib import pyplot as plt
import imutils
from rootsift import RootSIFT_create


def compute_similarity_with_descriptors(
    algorithm, image_a, image_b, presicion: float=50, norm: int=cv2.NORM_HAMMING
)->typing.Tuple[float, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    kp1, des1 = algorithm.detectAndCompute(image_a, None)
    kp2, des2 = algorithm.detectAndCompute(image_b, None)

    bf = cv2.BFMatcher(norm)
    # knnFilter results
    matches = bf.match(des1, des2)
    # View the maximum number of matching points
    good = [m for m in matches if m.distance < presicion]

    return (len(good) / len(matches), kp1, kp2, des1, des2, matches,)

def orb(
    image_a, image_b, presicion: float=50
)->typing.Tuple[float, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    return compute_similarity_with_descriptors(
        cv2.ORB_create(), image_a, image_b, presicion
    )

def brisk(
    image_a, image_b, presicion: float=50
)->typing.Tuple[float, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    return compute_similarity_with_descriptors(
        cv2.BRISK_create(), image_a, image_b, presicion
    )

def akaze(
    image_a, image_b, presicion: float=50
)->typing.Tuple[float, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    return compute_similarity_with_descriptors(
        cv2.AKAZE_create(), image_a, image_b, presicion
    )

def sift(
    image_a, image_b, presicion: float=50
)->typing.Tuple[float, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    return compute_similarity_with_descriptors(
        cv2.SIFT_create(), image_a, image_b, presicion, cv2.NORM_L2
    )

def root_sift(
    image_a, image_b, presicion: float=0.5
)->typing.Tuple[float, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    return compute_similarity_with_descriptors(
        RootSIFT_create(), image_a, image_b, presicion, cv2.NORM_L2
    )

def draw_features(
    image_a, image_b, key_points_a, key_points_b, features, number_of_features:int=100
):
    match_img = cv2.drawMatches(
        image_a, key_points_a, image_b, key_points_b, features[:number_of_features], None
    )
    cv2.imshow('Matches', match_img)
    cv2.waitKey()

def draw_differences_ssim(difference, image_a, image_b):
    diff = (difference * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image_a, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image_b, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # show the output images
    cv2.imshow("Original", image_a)
    cv2.imshow("Modified", image_b)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)