import glob
import os
import math
import random
from tkinter import BOTTOM, NW, RIGHT, SW, TOP, Button, Label, OptionMenu, Radiobutton, Scale, StringVar, Tk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps, ImageTk
from scipy import fftpack

laplacian = ([0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0])
gaussian = ([1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9])


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
            # ! non/zero division
            Iold = 1 if ((r+g+b)/3) == 0 else ((r+g+b)/3)
            Inew = Iold * value
            r = int(r*Inew/Iold)
            g = int(g*Inew/Iold)
            b = int(b*Inew/Iold)
            draw.point((i, j), (r, g, b))

    return outputimg

# note that input is image path


def BRF(ImageRGB):
    # ?: Open Images
    if Image.open(ImageRGB).mode == "RGB":
        R, G, B = Image.open(ImageRGB).resize((800, 800)).split()
        # ?: Convert input to array
        Rarr = np.array(R)
        Garr = np.array(G)
        Barr = np.array(B)
        # ?: Converting to frequency domain using fast fourier transform
        Rfft = fftpack.fftshift(fftpack.fft2(Rarr))
        Gfft = fftpack.fftshift(fftpack.fft2(Garr))
        Bfft = fftpack.fftshift(fftpack.fft2(Barr))
        # ?: BandReject filter
        x, y = Rarr.shape[0], Rarr.shape[1]

        # ? circle radius size L & R
        e_x, e_y = 50, 50  # // back ground frequency
        e2_x, e2_y = 25, 25  # // edges  frequency
        bbox = ((x/2)-(e_x/2), (y/2)-(e_y/2), (x/2)+(e_x/2), (y/2)+(e_y/2))
        bbox2 = ((x/2)-(e2_x/2), (y/2)-(e2_y/2),
                 (x/2)+(e2_x/2), (y/2)+(e2_y/2))

        bandpass = Image.new("L", (y, x), color=1)
        draw1 = ImageDraw.Draw(bandpass)
        draw1.ellipse(bbox, fill=0)
        draw1.ellipse(bbox2, fill=1)
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
    else:
        ImageL = Image.open(ImageRGB)
        imgarr = np.array(ImageL)
        imgfft = fftpack.fftshift(fftpack.fft2(imgarr))
        x, y = imgarr.shape[0], imgarr.shape[1]
        e_x, e_y = 85, 85  # // back ground frequency
        e2_x, e2_y = 45, 45  # // edges  frequency
        bbox = ((x/2)-(e_x/2), (y/2)-(e_y/2), (x/2)+(e_x/2), (y/2)+(e_y/2))
        bbox2 = ((x/2)-(e2_x/2), (y/2)-(e2_y/2),
                 (x/2)+(e2_x/2), (y/2)+(e2_y/2))
        bandpass = Image.new("L", (y, x), color=1)
        draw1 = ImageDraw.Draw(bandpass)
        draw1.ellipse(bbox, fill=0)
        draw1.ellipse(bbox2, fill=1)
        BParr = np.array(bandpass)
        plt.imshow(bandpass)
        plt.show()

        imgnew = np.multiply(imgfft, BParr)
        imgoutfft = fftpack.ifft2(fftpack.ifftshift(imgnew))
        OutImg = Image.new("L", (x, y))
        draw2 = ImageDraw.Draw(OutImg)
        for i in range(x):
            for j in range(y):
                val = int(abs(imgoutfft[i][j]))
                draw2.point((i, j), val)

    return OutImg.rotate(270.0).transpose(Image.FLIP_LEFT_RIGHT)

# note that input is image path


def Filter(inputimg, stride=1, padding=1, filter=laplacian):
    inputimg = Image.open(inputimg)
    if inputimg.mode == 'L':
        inputimg = ImageOps.grayscale(inputimg)

    pixel = inputimg.load()
    width, height = inputimg.size

    outimg = Image.new(inputimg.mode, (width, height))
    draw = ImageDraw.Draw(outimg)

    if inputimg.mode == 'RGB':

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

    elif inputimg.mode == 'L':

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
    return equalized_image


def Cluster_mean(clusters, centroids):
    for i in range(len(clusters)):
        r, g, b = 0, 0, 0
        for j in range(len(clusters[i])):
            color, xy = clusters[i][j]
            r += color[0]
            g += color[1]
            b += color[2]
        centroids[i][0] = r//j, g//j, b//j
    return centroids


def Calc_dest(arr3D, centroids, clusters):

    for i in range(0, arr3D.shape[0]):
        for j in range(0, arr3D.shape[1]):

            res = []
            resXY = []
            # ? calculate the difference foreeach point W\ centroids
            for center in centroids:
                color, x, y = center
                r, g, b = color
                res.append(int(math.sqrt(
                    int(math.pow(abs(r-arr3D[i][j][0]), 2)) +
                    int(math.pow(abs(g-arr3D[i][j][1]), 2)) +
                    int(math.pow(abs(b-arr3D[i][j][2]), 2))
                )))
                resXY.append([x, y])
            point = np.min(res)
            cluster_index = res.index(point)
            clusters[cluster_index].append([arr3D[i][j], [i, j]])
    centroids = Cluster_mean(clusters, centroids)
    return clusters, centroids


def k_means(arr3D, k=5):
    centroids = []
    clusters = []
    for i in range(0, k):
        clusters.append([])
    for i in range(0, k):
        x, y = random.randrange(
            0, arr3D.shape[0]), random.randrange(0, arr3D.shape[1])
        centroids.append([arr3D[x][y], x, y])

    newcentroids = []
    i = 0
    while i < 1:
        clusters, newcentroids = Calc_dest(arr3D, centroids, clusters)
        i += 1
        for m in range(len(clusters)):
            for j in range(len(clusters[m])):
                color, l, u = centroids[m]
                clusters[m][j] = color, clusters[m][j][1]

    return clusters, newcentroids


def Open_dialog():
    global imgpath, img, photo
    imgpath = filedialog.askopenfilename(
        title="Open Image", initialdir="./", filetypes=(("jpg files", "*.jpg"), ("All files", "*.*")))
    if imgpath != "":
        imglbl.config(text=imgpath)
        photo = Image.open(imgpath).resize((600, 600))
        img = ImageTk.PhotoImage(photo)
        photolbl.config(text="Input Image", image=img, compound=BOTTOM)


def execute_filter(event):
    global imgpath, img, photo
    if selected_filter.get() == "Laplacian filter":
        photo = Filter(imgpath, filter=laplacian)
        photo = photo.resize((600, 600))
    if selected_filter.get() == "Gaussian filter":
        photo = Filter(imgpath, filter=gaussian)
        photo = photo.resize((600, 600))
    messagebox.showinfo("Done", "Filter has been made into given Image")
    img = ImageTk.PhotoImage(photo)
    photolbl.config(text="Filtered Image", image=img, compound=BOTTOM)


def Bandrej():
    global imgpath, img, photo
    photo = BRF(imgpath)
    photo = photo.resize((600, 600))
    messagebox.showinfo("Done", "BandReject has been made into given Image")
    img = ImageTk.PhotoImage(photo)
    photolbl.config(text="After BandReject ", image=img, compound=BOTTOM)


def equalization():
    global imgpath, img, photo
    photo = Histogram_Equalization(imgpath)
    photo = photo.resize((600, 600))
    messagebox.showinfo(
        "Done", "Histogram Equalization has been made into given Image")
    img = ImageTk.PhotoImage(photo)
    photolbl.config(text="After Equalization ", image=img, compound=BOTTOM)


def means():
    global imgpath, img, photo
    temp = Image.open(imgpath)
    temparr = np.asarray(temp)
    clusters, centroids = k_means(temparr, 6)
    outimg = Image.new("RGB", temp.size, color=0)
    draw = ImageDraw.Draw(outimg)
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            color, xy = clusters[i][j]
            draw.point((xy[1], xy[0]), (color[0], color[1], color[2]))
    photo = outimg.resize((600, 600))
    messagebox.showinfo("Done", "K-means has been made into given Image")
    img = ImageTk.PhotoImage(photo)
    photolbl.config(text="After Segmentaion ", image=img, compound=BOTTOM)


def Iluminate(ratio):
    global imgpath, img, photo
    photo = Brightniess(imgpath, ratio)
    photo = photo.resize((600, 600))
    messagebox.showinfo("Done", "Image Brightness has been modified")
    img = ImageTk.PhotoImage(photo)
    photolbl.config(text="After Brightness Modification ",
                    image=img, compound=BOTTOM)


if __name__ == "__main__":

    root = Tk()
    root.title("Image Processing App")

    root.minsize(500, 500)
    root.resizable(width=False, height=False)

    opnbtn = Button(root, text="Open Image")
    imgpath = ""

    opnbtn.config(command=Open_dialog)

    imglbl = Label(root, text=imgpath)
    photolbl = Label(root, width=600, height=600)
    Open_dialog()

    opnbtn.pack(side=TOP, anchor=NW, padx=10, pady=10)
    if imgpath != "":
        photolbl.pack(side=RIGHT, anchor=NW, pady=10, padx=10)
        imglbl.pack(side=BOTTOM, anchor=SW, padx=10, pady=10)
    filtervar = 0
    bandrejectvar = 0
    histequalvar = 0
    kmvar = 0
    brightvar = 0
    histovar = 0
    selected_filter = StringVar()
    # Select filter then pass to the dunction     # ? Done
    r1 = Radiobutton(root, text="Filters", variable=filtervar, value=0)
    # Radio button check function calls only  # ? Done
    r2 = Radiobutton(root, text="Band Reject",
                     variable=bandrejectvar, value=1, command=Bandrej)
    r3 = Radiobutton(root, text="Histogram Equalization", variable=histequalvar, value=2,
                     command=equalization)  # Radio button check function calls only  # ? Done
    # Radio button check function calls only  # ? Done
    r4 = Radiobutton(root, text="K-means", variable=kmvar,
                     value=3, command=means)
    r5 = Radiobutton(root, text="Brightness", variable=brightvar, value=4, command=lambda: Iluminate(
        slider.get()))  # Slider passing the brightness value to the function  # ? Done
    # Radio button check only calls the function
    r6 = Radiobutton(root, text="Histogram", variable=histovar, value=5)
    r1.select()

    cmbbx1 = OptionMenu(root, selected_filter, "Laplacian filter",
                        "Gaussian filter", command=execute_filter)
    global slider
    slider = Scale(root, from_=0, to=2, orient="horizontal", resolution=0.1)
    brightvar = slider.get()
    r1.pack(side=TOP, anchor=NW)
    cmbbx1.pack(side=TOP, anchor=NW)
    r2.pack(side=TOP, anchor=NW)
    r3.pack(side=TOP, anchor=NW)
    r4.pack(side=TOP, anchor=NW)
    r5.pack(side=TOP, anchor=NW)
    slider.pack()
    r6.pack(side=TOP, anchor=NW)
    root.mainloop()
    """
    for img in imgs:
        pass
        # TODO:: Histogram show
        # ! Save_Images(OutImages,"Image_name")
    """
