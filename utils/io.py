import numpy as np
def read_bin(filename):
    with open(filename,'r') as f:
        width, height = np.fromfile(f, dtype = np.uint32, count=2)
        f.seek(4*2)
        image = np.fromfile(f, dtype = np.float32)
        image = image.reshape((width, height))
        return image 

def write_bin(image, filename):
    size = np.array([image.shape[0],image.shape[1]],dtype=np.uint32)
    image = image.astype(np.float32).reshape(-1)
    with open(filename, 'wb') as f:
        f.write(size[1])
        f.write(size[0])
        image.tofile(f)
