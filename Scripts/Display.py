from IPython.display import Image, display
import glob
images = glob.glob('C:/Users/mengbo/PycharmProjects/Advanced-ALPR-System/Scripts/runs/detect/predict/*.jpg')#where the result save
for image in images:
   display(Image(filename = image,width=400))