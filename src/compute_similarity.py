from skimage.metrics import structural_similarity as ssim
from image_similarity_measures.quality_metrics import fsim, rmse, psnr, sre, issm
from sewar.full_ref import msssim, scc
import ssim as pyssim
import cv2
from utils import draw_differences_ssim, orb, sift, akaze, brisk, draw_features, root_sift


absolute_path_imageA = "/home/rainyel/Documents/datamart/image-comparison/src/output/image_a.jpg"
absolute_path_imageB = "/home/rainyel/Documents/datamart/image-comparison/src/output/image_b.jpg"

imageA = cv2.imread(absolute_path_imageA)
imageB = cv2.imread(absolute_path_imageB)

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# fsim_value = fsim(imageA, imageB)
scc_value = scc(grayA, grayB)
issm_value = issm(grayA, grayB) # FIXME - it is not working, return 0.0
rmse_value = rmse(imageA, imageB) 
psnr_value = psnr(imageA, imageB) 
sre_value = sre(imageA, imageB)
ssim_value = ssim(grayA, grayB, full=True)
msssim_value = msssim(grayA, grayB)
# dssim_value = (1-ssim_value[0]) / 2
orb_value = orb(grayA, grayB)
brisk_value = brisk(grayA, grayB, 90)
akaze_value = akaze(grayA, grayB)
sift_value = sift(grayA, grayB)
root_sift_value = root_sift(grayA, grayB, presicion=0.24) 

# NOTE - the process is killed
# cw_ssim_value = pyssim.SSIM(absolute_path_imageA).cw_ssim_value(absolute_path_imageB)


print(f"SSIM: {ssim_value[0]}")
print(f"MSSSIM: {msssim_value}")
# print(f"DSSIM: {dssim_value}")
print(f"RMSE: {rmse_value}")
print(f"PSNR: {psnr_value}")
print(f"SRE: {sre_value}")
print(f"ISSM: {issm_value}")
print(f"SCC: {scc_value}")
print(f"ORB: {orb_value[0]}")
print(f"BRISK: {brisk_value[0]}")
print(f"AKAZE: {akaze_value[0]}")
print(f"SIFT: {sift_value[0]}")
print(f"RootSIFT: {root_sift_value[0]}")
# print(f"CW-SSIM: {cw_ssim_value}")
# print(f"FSIM: {fsim_value}")

# draw_features(grayA, grayB, orb_value[1], orb_value[2], orb_value[-1])
# draw_features(
#     grayA, grayB, brisk_value[1], brisk_value[2], 
#     sorted(brisk_value[-1], key=lambda x: x.distance), 999
# )
# draw_features(
#     grayA, grayB, sift_value[1], sift_value[2], 
#     sorted(sift_value[-1], key=lambda x: x.distance), 999
# )

# draw_differences_ssim(ssim_value[1], grayA, grayB)