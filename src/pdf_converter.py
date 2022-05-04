import pdf2image
import os

absolute_path = "/home/rainyel/Documents/datamart/image-comparison/src/"

pdf_absolute_path = absolute_path + "pdf"
output_absolute_path = absolute_path + "output"

for pdf in os.listdir(pdf_absolute_path):
    print(pdf + "\n")

    pdf_path = output_absolute_path + "/" + pdf
    os.makedirs(pdf_path)
    images_from_path = pdf2image.convert_from_path(
        pdf_absolute_path+"/"+pdf, output_folder=pdf_path
    )
