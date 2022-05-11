from skimage.metrics import structural_similarity as ssim
import cv2
from image_similarity_measures.quality_metrics import fsim, rmse, psnr, sre, issm
from sewar.full_ref import msssim, scc
import ssim as pyssim
from matplotlib import pyplot as plt

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

# descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(imageA, None)
kp2, des2 = orb.detectAndCompute(imageB, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

            # knnFilter results
matches = bf.match(des1, des2)

            # View the maximum number of matching points
good = [m for m in matches if m.distance < 90]
print(len(good))
print(len(matches))
similary = len(good) / len(matches)
print(similary)
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
# print(f"CW-SSIM: {cw_ssim_value}")
# print(f"FSIM: {fsim_value}")

