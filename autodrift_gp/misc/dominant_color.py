import cv2
import numpy as np
from skimage import io
import webcolors
import matplotlib.pyplot as plt

def main():
    # M x N x 4 for rgba images
    img = io.imread('https://i.stack.imgur.com/DNM65.png')
    print(img.shape)
    # print(img[:, :, 0:2])    # get all but last (alpha)
    img = img[:, :, 0:3]
    print(img.shape)

    # get average colors
    average = img.mean(axis=0).mean(axis=0)
    print(average)

    # get dominant color
    # You can think of reshaping as first raveling the array (using the given index order), then inserting the elements from the raveled array into the new array using the same kind of index ordering as was used for the raveling.
    pixels = np.float32(img.reshape(-1, 3))
    print(pixels.shape)

    # cluster image pixels into 5 (color) clusters, 0 to 4
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    print(labels.shape) # clustering output
    print("Palette shape {0}".format(palette.shape))
    print(palette)

    _, counts = np.unique(labels, return_counts=True) # (ndarray) return the number of times each unique item appears in ar.
    print(counts)
    dominant = palette[np.argmax(counts)]
    print(dominant)

    # get dominant colors, by color type
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices] / counts.sum()]))
    rows = np.int_(img.shape[0] * freqs)

    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    print(dom_patch)

    avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average)

    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices] / counts.sum()]))
    rows = np.int_(img.shape[0] * freqs)

    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    print(palette)
    print(indices) # indices of palette colors in decreasing level of dominance

    # print colors in terms of dominant to least dominant
    for i in range(len(indices)):
        idx = indices[i]
        requested_color = palette[idx][0], palette[idx][1], palette[idx][2]
        actual_name, closest_name = get_colour_name(requested_color)
        print("Actual colour name: {0}, closest color name: {1}".format(actual_name, closest_name))

# https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python
def closest_colour(requested_colour):
    min_colours = {}
    print(webcolors.CSS2_HEX_TO_NAMES)
    for key, name in webcolors.CSS2_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

if __name__ == '__main__':
    main()