import pdf2image
import os

absolute_path = "/home/rainyel/Documents/datamart/image-comparison/src/"

pdf_absolute_path = absolute_path + "pdf"
output_absolute_path = absolute_path + "output"

for pdf in os.listdir(pdf_absolute_path):
    print(pdf + "\n")

    pdf_path = output_absolute_path + "/" + pdf.replace(".pdf", "")
    os.makedirs(pdf_path)
    # REVIEW - use the same size to apply comparision????
    images_from_path = pdf2image.convert_from_path(
        pdf_absolute_path+"/"+pdf, output_folder=pdf_path
    )

# For instance, PSNR, RMSE, or SRE simply measure how different the two images are. 
# This is good to make sure that a predicted or restored image is similar to its 
# "target" image, but the metics don't consider the quality of the image itself. 
# Other metrics attempt to solve this problem by considering image structure (SSIM) 
# or displayed features (FSIM).
# The PSNR does not correlate well with perceived visual
# Quality