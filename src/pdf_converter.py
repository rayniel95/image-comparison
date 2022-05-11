import math
from numpy import mat
import pdf2image
from PIL import Image

absolute_path = "/home/rainyel/Documents/datamart/image-comparison/src/"

pdf_absolute_path = absolute_path + "pdf"
output_absolute_path = absolute_path + "output"

absolute_path_a = "/home/rainyel/Documents/datamart/image-comparison/src/pdf/8748fc1c-132d-4c3a-88fc-1c132d0c3a38.pdf"
absolute_path_b = "/home/rainyel/Documents/datamart/image-comparison/src/pdf/33_cristal report.pdf"

absolute_path_output = "/home/rainyel/Documents/datamart/image-comparison/src/output/"

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

image_a = pdf2image.convert_from_path(
    absolute_path_a, fmt="jpeg", 
    output_file=absolute_path_a.split("/")[-1].replace(".pdf", ""), thread_count=5, 
    grayscale=True
)

image_b = pdf2image.convert_from_path(
    absolute_path_b, fmt="jpeg",
    output_file=absolute_path_b.split("/")[-1].replace(".pdf", ""), thread_count=5, 
    grayscale=True
)

max_height = max(image_a[0].size[1], image_b[0].size[1])
max_width = max(image_a[0].size[0], image_b[0].size[0])

image_a_resized = add_margin(
    image_a[0], max_height - image_a[0].size[1], max_width - image_a[0].size[0], 0, 
    0, "white"
)

image_b_resized = add_margin(
    image_b[0], max_height - image_b[0].size[1], max_width - image_b[0].size[0], 0, 
    0, "white"
)

image_a_resized.save(absolute_path_output+"image_a.jpg", "jpeg")
image_b_resized.save(absolute_path_output+"image_b.jpg", "jpeg")

# For instance, PSNR, RMSE, or SRE simply measure how different the two images are. 
# This is good to make sure that a predicted or restored image is similar to its 
# "target" image, but the metics don't consider the quality of the image itself. 
# Other metrics attempt to solve this problem by considering image structure (SSIM) 
# or displayed features (FSIM).
# The PSNR does not correlate well with perceived visual
# Quality