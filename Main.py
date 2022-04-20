from __future__ import division
import glob, os, numpy as np, matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from scipy import fftpack

laplacian = ([0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0])

def Load_images():
    Images = []
    for infile in glob.glob("./InputImage/*.jpg"):
        file, ext = os.path.splitext(infile)
        Images.append(infile)
    return Images

def Save_Images(lst,name):
    k = 0
    for element in lst:
        element.save(f'./OutputImage/_{name}_{k}.png')
        k += 1

def Brightniess(rgbimg, value):
    
    input_pixels = rgbimg.load()
    width, height = rgbimg.size
    # creating new image for outputting
    outputimg = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(outputimg)

    for i in range(width):
        for j in range(height):
            r, g, b = input_pixels[i, j]
            Iold = ((r+g+b)/3)+1          ##! non/zero division
            Inew = Iold * value
            r = int(r*Inew/Iold)
            g = int(g*Inew/Iold)
            b = int(b*Inew/Iold)
            draw.point((i, j), (r, g, b))

    return outputimg

def BRF(ImageRGB):
    # TODO: Open Images
    R, G, B = Image.open(ImageRGB).resize((800, 800)).split()
    # TODO: Convert input to array
    Rarr = np.array(R)
    Garr = np.array(G)
    Barr = np.array(B)
    # TODO: Converting to frequency domain using fast fourier transform
    Rfft = fftpack.fftshift(fftpack.fft2(Rarr))
    Gfft = fftpack.fftshift(fftpack.fft2(Garr))
    Bfft = fftpack.fftshift(fftpack.fft2(Barr))
    # TODO: BandReject filter
    x, y = Rarr.shape[0], Rarr.shape[1]
    print(x, y)
    # ? circle radius size L & R
    e_x, e_y = 250, 250  # // back ground frequency
    e2_x, e2_y = 205, 205  # // edges  frequency
    bbox = ((x/2)-(e_x/2), (y/2)-(e_y/2), (x/2)+(e_x/2), (y/2)+(e_y/2))
    bbox2 = ((x/2)-(e2_x/2), (y/2)-(e2_y/2), (x/2)+(e2_x/2), (y/2)+(e2_y/2))
    print(bbox)
    print(bbox2)
    bandpass = Image.new("L", (y, x), color=1)
    draw1 = ImageDraw.Draw(bandpass)
    draw1.ellipse(bbox, fill=1)
    draw1.ellipse(bbox2, fill=0)
    BParr = np.array(bandpass)
    plt.imshow(bandpass)
    plt.show()
    Rnew = np.multiply(Rfft, BParr)
    Gnew = np.multiply(Gfft, BParr)
    Bnew = np.multiply(Bfft, BParr)

    Routfft = fftpack.ifft2(fftpack.ifftshift(Rnew))
    Goutfft = fftpack.ifft2(fftpack.ifftshift(Gnew))
    Boutfft = fftpack.ifft2(fftpack.ifftshift(Bnew))
    OutImg = Image.new("RGB", (x, y))
    draw2 = ImageDraw.Draw(OutImg)
    for i in range(x):
        for j in range(y):
            r = int(abs(Routfft[i][j]))
            g = int(abs(Goutfft[i][j]))
            b = int(abs(Boutfft[i][j]))
            draw2.point((i, j), (r, g, b))

    return OutImg.rotate(270.0)

def Filter(inputimg, stride=1, padding=1, filter=laplacian, Type='RGB'):
    
    if Type == 'L':
        inputimg = ImageOps.grayscale(inputimg)

    pixel = inputimg.load()
    width, height = inputimg.size

    outimg = Image.new(Type, (width, height))
    draw = ImageDraw.Draw(outimg)

    if Type == 'RGB':

        for x in range(padding, width-padding):
            for y in range(padding, height-padding):
                res = [0, 0, 0]
                for a in range(len(filter)):
                    for b in range(len(filter)):
                        xrow = x+a-padding
                        ycol = y+b-padding
                        res[0] += int((pixel[xrow, ycol][0] * filter[a][b]))
                        res[1] += int((pixel[xrow, ycol][1] * filter[a][b]))
                        res[2] += int((pixel[xrow, ycol][2] * filter[a][b]))

                draw.point((x, y), (int(res[0]), int(res[1]), int(res[2])))
                y += stride
            x += stride

        return outimg

    elif Type == 'L':

        for x in range(padding, width-padding):
            for y in range(padding, height-padding):
                res = 0
                for a in range(len(filter)):
                    for b in range(len(filter)):
                        xrow = x+a-padding
                        ycol = y+b-padding
                        res += int((pixel[xrow, ycol] * filter[a][b]))

                draw.point((x, y), res)
                y += stride
            x += stride

        return outimg


if __name__ == "__main__":
    imgs = Load_images()
    OutImages = []
    for img in imgs:
        pass
        ##?NOTE::function call only 
        ##TODO:: K-means
        ##// Band reject
        ##? Done
        ##TODO:: Histogram equalization
        ##// Filter
        ##? Done
        ##// Brightness
        ##? Done
        ##TODO:: Histogram show 
    ##! Save_Images(OutImages,"Image_name")  