from utils import io, mapreduceKnnRemoval
import time

if __name__ == "__main__":
    # param
    filename = "./data/cropped.bin"
    outfilename = "./data/test_out.bin"
    k = 30
    kernelSize = 21
    dist_th = 1.5
    # load image
    image = io.read_bin(filename)
    # knnRemoval
    st = time.time()
    filtered, count = mapreduceKnnRemoval.knnRemoval(image, k, kernelSize, dist_th)
    ed = time.time() - st
    # restore image
    io.write_bin(filtered, outfilename)
    print("knnRemoval takes %f s, removes %d pixels"%(ed, count))
