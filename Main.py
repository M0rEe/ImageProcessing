import glob
import os
import numpy as np
import matplotlib.pyplot as plt
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


def Save_Images(lst, name):
    k = 0
    for element in lst:
        element.save(f'./OutputImage/_{name}_{k}.png')
        k += 1

# note that input is image path


def Brightniess(rgbimg, value):
    rgbimg = Image.open(rgbimg)
    input_pixels = rgbimg.load()
    width, height = rgbimg.size
    # creating new image for outputting
    outputimg = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(outputimg)

    for i in range(width):
        for j in range(height):
            r, g, b = input_pixels[i, j]
            Iold = ((r+g+b)/3)+1  # ! non/zero division
            Inew = Iold * value
            r = int(r*Inew/Iold)
            g = int(g*Inew/Iold)
            b = int(b*Inew/Iold)
            draw.point((i, j), (r, g, b))

    return outputimg

# note that input is image path


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

    # ? circle radius size L & R
    e_x, e_y = 250, 250  # // back ground frequency
    e2_x, e2_y = 205, 205  # // edges  frequency
    bbox = ((x/2)-(e_x/2), (y/2)-(e_y/2), (x/2)+(e_x/2), (y/2)+(e_y/2))
    bbox2 = ((x/2)-(e2_x/2), (y/2)-(e2_y/2), (x/2)+(e2_x/2), (y/2)+(e2_y/2))

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

# note that input is image path


def Filter(inputimg, stride=1, padding=1, filter=laplacian, Type='RGB'):
    inputimg = Image.open(inputimg)
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

# note that input is image path


def Histogram_Equalization(img_path):
    # convert to grayScale
    Input_Image = Image.open(img_path)
    Input_image_grayScale = ImageOps.grayscale(Input_Image)
    Input_image_grayScale_pixels = Input_image_grayScale.load()
    width, height = Input_Image.size

    # Visualize the histogram of original Image
    Histogram_frequency = Input_image_grayScale.histogram()
    Histogram_index = np.arange(256)
    plt.bar(x=Histogram_index, height=Histogram_frequency)
    plt.show(block=False)

    # get probability of each pixel
    total_number_pixels = width * height
    Probability = []
    for x in range(0, 256):
        if Histogram_frequency[x] == 0:
            Histogram_frequency[x] = 1
        Probability.append(Histogram_frequency[x] / total_number_pixels)

    # Equalizer the image
    changepixel = {}
    last_sum = 0
    for x in range(256):
        last_sum = last_sum + (255 * Probability[x])
        changepixel[x] = (math.floor(last_sum))

    equalized_image = Image.new("L", Input_Image.size)
    draw = ImageDraw.Draw(equalized_image)
    for x in range(0, Input_image_grayScale.width - 1):
        for y in range(0, Input_image_grayScale.height - 1):
            draw.point(
                (x, y), (int(changepixel.get(Input_image_grayScale_pixels[x, y]))))

    # histogram of New_Image
    Histogram_frequency = equalized_image.histogram()

    Histogram_index = np.arange(256)
    plt.bar(x=Histogram_index, height=Histogram_frequency)
    plt.show()
    # show the equalized Image
    equalized_image.show()
    return equalized_image


if __name__ == "__main__":
    imgs = Load_images()
    OutImages = []
    for img in imgs:
        pass
        # ?NOTE::function call only
        # TODO:: K-means
        # // Band reject
        # ? Done
        # // Histogram equalization
        # ? Done
        # // Filter
        # ? Done
        # // Brightness
        # ? Done
        # TODO:: Histogram show
    # ! Save_Images(OutImages,"Image_name")
