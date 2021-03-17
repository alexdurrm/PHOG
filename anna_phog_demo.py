from anna_phog import anna_phog
import imageio
import matplotlib.pyplot as plt


image_path = "image_0058.jpg"
S = 8
angle = 360
Level = 3
roi = [1,225,1,300]
save=True

Image = imageio.imread(image_path)
p = anna_phog(Image, bin, angle, Level, roi)
print("P: \n{}".format(p))
print(len(p), type(p))
