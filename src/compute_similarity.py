from skimage.metrics import structural_similarity as ssim
import cv2
from image_similarity_measures.quality_metrics import fsim, rmse, psnr, sre, issm
from sewar.full_ref import msssim, scc
import ssim as pyssim
from matplotlib import pyplot as plt
import typing


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
    # draw first 50 matches
    match_img = cv2.drawMatches(
        image_a, key_points_a, image_b, key_points_b, features[:number_of_features], None
    )
    cv2.imshow('Matches', match_img)
    cv2.waitKey()

absolute_path_imageA = "/home/rainyel/Documents/datamart/image-comparison/src/output/image_a.jpg"
absolute_path_imageB = "/home/rainyel/Documents/datamart/image-comparison/src/output/image_b.jpg"

imageA = cv2.imread(absolute_path_imageA)
imageB = cv2.imread(absolute_path_imageB)

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# fsim_value = fsim(imageA, imageB)
scc_value = scc(grayA, grayB)
issm_value = issm(imageA, imageB) # FIXME - it is not working, return 0.0
rmse_value = rmse(imageA, imageB) 
psnr_value = psnr(imageA, imageB) 
sre_value = sre(imageA, imageB)
ssim_value = ssim(grayA, grayB)
msssim_value = msssim(grayA, grayB)
dssim_value = (1-ssim_value) / 2
orb_value = orb(grayA, grayB)
brisk_value = brisk(grayA, grayB, 90)
akaze_value = akaze(grayA, grayB)
sift_value = sift(grayA, grayB)

# NOTE - the process is killed
# cw_ssim_value = pyssim.SSIM(absolute_path_imageA).cw_ssim_value(absolute_path_imageB)


print(f"SSIM: {ssim_value}")
print(f"MSSSIM: {msssim_value}")
print(f"DSSIM: {dssim_value}")
print(f"RMSE: {rmse_value}")
print(f"PSNR: {psnr_value}")
print(f"SRE: {sre_value}")
print(f"ISSM: {issm_value}")
print(f"SCC: {scc_value}")
print(f"ORB: {orb_value[0]}")
print(f"BRISK: {brisk_value[0]}")
print(f"AKAZE: {akaze_value[0]}")
print(f"SIFT: {sift_value[0]}")
# print(f"CW-SSIM: {cw_ssim_value}")
# print(f"FSIM: {fsim_value}")

# draw_features(grayA, grayB, orb_value[1], orb_value[2], orb_value[-1])
# draw_features(
#     grayA, grayB, brisk_value[1], brisk_value[2], 
#     sorted(brisk_value[-1], key=lambda x: x.distance), 999
# )
draw_features(
    grayA, grayB, sift_value[1], sift_value[2], 
    sorted(sift_value[-1], key=lambda x: x.distance), 999
)