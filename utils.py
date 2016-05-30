import numpy as np

def color_image(image):
    import matplotlib.cm as cm
    mycm = cm.get_cmap('Set1')
    return mycm(image)