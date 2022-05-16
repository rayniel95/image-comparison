import cv2
import typing
from matplotlib import pyplot as plt


def compute_similarity_with_descriptors(
    algorithm, image_a, image_b, presicion: int=50, norm: int=cv2.NORM_HAMMING
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
    image_a, image_b, presicion: int=50
)->typing.Tuple[float, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    return compute_similarity_with_descriptors(
        cv2.ORB_create(), image_a, image_b, presicion
    )

def brisk(
    image_a, image_b, presicion: int=50
)->typing.Tuple[float, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    return compute_similarity_with_descriptors(
        cv2.BRISK_create(), image_a, image_b, presicion
    )

def akaze(
    image_a, image_b, presicion: int=50
)->typing.Tuple[float, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    return compute_similarity_with_descriptors(
        cv2.AKAZE_create(), image_a, image_b, presicion
    )

def sift(
    image_a, image_b, presicion: int=50
)->typing.Tuple[float, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    return compute_similarity_with_descriptors(
        cv2.SIFT_create(), image_a, image_b, presicion, cv2.NORM_L2
    )

def draw_features(
    image_a, image_b, key_points_a, key_points_b, features, number_of_features:int=100
):
    match_img = cv2.drawMatches(
        image_a, key_points_a, image_b, key_points_b, features[:number_of_features], None
    )
    cv2.imshow('Matches', match_img)
    cv2.waitKey()