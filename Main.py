import glob,os


def Load_images():
    Images = []
    for infile in glob.glob("./InputImage/*.jpg"):
        file, ext = os.path.splitext(infile)
        Images.append(infile)
    return Images

def Save_Images(lst):
    k = 0
    for element in lst:
        element.save(f'./OutputImage/_BPF_{k}.png')
        k += 1



if __name__ == "__main__":
    imgs = Load_images()
    OutImages = []
    for img in imgs:
        pass
        ##?NOTE::function call only 
        ##TODO:: K-means
        ##TODO:: Band reject
        ##TODO:: Histogram equalization
        ##TODO:: Filter
        ##TODO:: Brightness
        ##TODO:: Histogram show 
    ##! Save_Images(OutImages)  