from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import webcolors


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = webcolors.rgb_to_name(requested_colour)
    except:
        closest_name = closest_colour(requested_colour)
    return closest_name


def show_img_compar(img_1, img_2):
    f, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis("off")  # hide the axis
    ax[1].axis("off")
    f.tight_layout()
    plt.show()


def palette(clusters):
    palette = np.zeros((1, clusters.cluster_centers_.shape[0], 3), np.uint8)
    steps = 1
    for idx, centers in enumerate(clusters.cluster_centers_):
        palette[:, int(idx * steps) : (int((idx + 1) * steps)), :] = centers
    return palette


def get_colors(img_path):
    im = Image.open(img_path)
    w, h = im.size

    im_cropped = im.crop((w // 3, h // 8, 2 * w // 3, h))
    # im_cropped = im
    img = np.asarray(im_cropped)

    # img = cv.imread("ss3.jpg")
    # print(type(img))
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # img_2 = cv.imread("ss2.png")
    # img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)

    dim = (500, 300)
    # resize image
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    # img_2 = cv.resize(img_2, dim, interpolation = cv.INTER_AREA)

    clt = KMeans(n_clusters=3)

    clt_1 = clt.fit(img.reshape(-1, 3))
    img_palette = palette(clt_1)[0]

    darkest_color = (
        int(img_palette[0][0]),
        int(img_palette[0][1]),
        int(img_palette[0][2]),
    )

    for i in img_palette:
        if darkest_color[0] + darkest_color[1] + darkest_color[2] > int(i[0]) + int(
            i[1]
        ) + int(i[2]):
            darkest_color = (int(i[0]), int(i[1]), int(i[2]))

    color_name = get_colour_name(darkest_color)

    return color_name

    # show_img_compar(img, palette(clt_1))
    # clt_2 = clt.fit(img_2.reshape(-1, 3))
    # show_img_compar(img_2, palette(clt_2))


if __name__ == "__main__":
    print(get_colors("a.png"))
