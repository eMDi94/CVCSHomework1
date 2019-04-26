from parameters import *

import numpy as np
from scipy.spatial import distance as dist
import cv2

RECTANGLE_SECTOR_TL = 0
RECTANGLE_SECTOR_TR = 1
RECTANGLE_SECTOR_BR = 2
RECTANGLE_SECTOR_BL = 3

def linear_stretch(img, a, b):
    """
    Stretch the image
    :param img: 2D ndarray representing an image plane
    :param a: scale factor
    :param b: sum factor
    :return: The stretched image
    """
    img = img.astype(np.float)
    img = np.round(img * a + b)
    img[img < 0.] = 0.
    img[img > 255.] = 255.
    return img.astype(np.uint8)


def labels_map_to_bgr(labels_map):
    """
    Maps a label map to a bgr image.
    :param labels_map: labels map
    :return: bgr image
    """
    labels_hue = np.uint8(179 * labels_map / labels_map.max())
    blank_ch = 255 * np.ones_like(labels_hue)
    labeled_img = cv2.merge((labels_hue, blank_ch, blank_ch))
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[labels_hue == 0] = np.zeros(3, dtype=np.uint8)
    return labeled_img


def show(img, winname = 'out'):
    """
    Show an image
    :param img: image to show
    :param winname: windows name
    :return:
    """
    cv2.imshow(winname, img)
    cv2.waitKey()


def resize_to_ratio(img, ratio):
    """
    Resize an image according to the given ration
    :param img: Image to be resized
    :param ratio: ratio used to resize the image
    :return: Image resized
    """
    assert ratio > 0, 'ratio_percent must be > 0'
    w = int(img.shape[1] * ratio)
    h = int(img.shape[0] * ratio)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def compute_histogram(img):
    """
    Compute the histogram of the image
    :param img: 2D or 3D array
    :return: normalized histogram
    """
    assert len(img.shape) >= 2, 'img.shape not valid'
    if len(img.shape) == 3:
        h, w, d = img.shape
        h_w = h * w
        if d == 3:
            p1 = img[:, :, 0]
            p2 = img[:, :, 1]
            p3 = img[:, :, 2]
            planes = [p1, p2, p3]
        else:
            planes = [img]

    if len(img.shape) == 2:
        h_w, d = img.shape
        if d == 3:
            p1 = img[:, 0]
            p2 = img[:, 1]
            p3 = img[:, 2]
            planes = [p1, p2, p3]
        else:
            planes = [img]

    histogram = np.zeros(256*d) # e' corretto 256, non h_w
    for i in np.arange(len(planes)):
        p = planes[i]
        for val in np.unique(p):
            count = np.sum(p == val)
            histogram[val + i*256] = count
    histogram = histogram / img.size
    return histogram


def convert_to(img, flag):
    """
    Convert the image into the given format
    :param img: Image to be converted
    :param flag: Flag pointing to the conversion type
    :return: Converted Image
    """
    return cv2.cvtColor(img, flag)


def entropy(histogram):
    """
    Compute Shannon's Entropy
    :param histogram: Histogram approximation of the real distribution
    :return: Shannon's Entropy
    """
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))


def remove_small_connectedComponents(labeled_img, threshold):
    """
    Function used to remove the small components
    :param labeled_img: Input image with components
    :param threshold: Area threshold
    :return: Image without the small components
    """
    img = np.array(labeled_img)
    labels, counts = np.unique(img, return_counts=True)
    for label, count in zip(labels, counts):
        if count < threshold:
            img[img == label] = 0
    return img


def sort_corners(corners):
    """
    Sort corners starting from top left one and going clockwise
    :param corners:
    :return: sorted corners
    """
    # trovo il "centro"
    c = find_intersection(corners[0], corners[2], corners[1], corners[3])
    if not (corners[0][0] < c[0] < corners[2][0]):
        c = find_intersection(corners[0], corners[1], corners[2], corners[3])
    # calcolo l'angolo con ogni punto del rettangolo
    a0 = find_angle_with_horizontal(c, corners[0])
    a1 = find_angle_with_horizontal(c, corners[1])
    a2 = find_angle_with_horizontal(c, corners[2])
    a3 = find_angle_with_horizontal(c, corners[3])
    print(np.argsort([a0, a1, a2, a3]))
    print(corners[np.argsort([a0, a1, a2, a3]), :])
    return corners[np.argsort([a0, a1, a2, a3]), :]


# TODO TMP REMOVE --start
def OLD_sort_corners(corners):
    # sort points based on their x
    xSorted = corners[np.argsort(corners[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
# TODO TMP REMOVE --FINE


def find_distance_squared(p1, p2):
    """
    :param a: [p1x, p1y]
    :param b: [p2x, p2y]
    :return: Scalar. distance between p1 and p2
    """
    assert len(p1) == 2, 'len(p1) must be equal to 2'
    assert len(p2) == 2, 'len(p2) must be equal to 2'
    return ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)


def find_distance(a, b):
    """
    :param a: [Ax, Ay]
    :param b: [Bx, By]
    :return: Scalar. distance between A and B
    """
    return np.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def find_angle_with_horizontal(a, b, length=None, use_negative=False):
    """
    :param a: [Ax, Ay]
    :param b: [Bx, By]
    :param length: Scalar. if not provided, will be calculated
    :param use_negative: True/False. When set to True, instead of angles greater than PI, negative values will be returned
    :return: Scalar. angle in radiant
    """
    if length is None:
        length = find_distance(a, b)
    assert length > 0, 'distance between points should be greater than 0'
    angle = np.arccos((b[0] - a[0]) / length)
    if a[1] > b[1]:
        if use_negative is True:
            angle = -angle
        else:
            angle = 2*np.pi - angle
    return angle


def radiant_to_degree(angle):
    return angle * 180 / np.pi


def stretch_considering_perspective(length, angle1, angle2, k=1):
    """
    :return: stretched length
    """
    assert angle1 >= 0, 'angle1 value should be positive. use absolute value?'
    assert angle2 >= 0, 'angle2 value should be positive. use absolute value?'
    if angle1 > np.pi / 2:
        angle1 = np.pi - angle1
    if angle2 > np.pi / 2:
        angle2 = np.pi - angle2
    angle_tot = angle1 + angle2
    ratio = np.abs(angle_tot / np.pi) * k
    stretched = length * (1 + ratio)
    #print("length: {}\nangle1: {}°\nangle2: {}°\nk: {}\nangle_tot: {}°\nratio: {}\nstretched: {}\n\n".format(length,radiant_to_degree(angle1),radiant_to_degree(angle2),k,radiant_to_degree(angle_tot),ratio,stretched))
    return stretched

# TODO REMOVE THIS ONE
def OLD_rectify_image(img, corners):
    # https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
    # Compute the width of the new image
    corners = corners.astype(np.float32)
    (tl, tr, br, bl) = corners
    width_bottom = find_distance(br, bl)  # np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_top = find_distance(tr, tl)  # np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # Compute the height of the new image
    height_right = find_distance(tr, br)  # np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_left = find_distance(tl, bl)  # np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    if width_bottom == 0 or width_top == 0 or height_right == 0 or height_left == 0:
        return None
    # Take the maximum of the width and height values to reach final dimensions
    maxWidth = max(int(width_bottom), int(width_top))
    maxHeight = max(int(height_right), int(height_left))
    # Find angles in respect to horizontal
    angle_bottom = find_angle_with_horizontal(br, bl, length=width_bottom, use_negative=True)
    angle_top = find_angle_with_horizontal(tr, tl, length=width_top, use_negative=True)
    angle_left = find_angle_with_horizontal(bl, tl, length=height_left, use_negative=True)
    angle_right = find_angle_with_horizontal(br, tr, length=height_right, use_negative=True)
    # adjust maxWidth considering perspective warping
    maxWidth = int(stretch_considering_perspective(maxWidth, np.abs(angle_bottom), np.abs(angle_top)))
    maxHeight = int(stretch_considering_perspective(maxHeight, np.abs(angle_left - np.pi / 2),
                                                               np.abs(angle_right - np.pi / 2)))
    # Construct destination points which will be used to map the screen to a top-down,
    final = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = np.float32)
    mat = cv2.getPerspectiveTransform(corners, final)
    # Apply the transformation on the first image using cv2.warpPerspective()
    dst = cv2.warpPerspective(img, mat, (maxWidth, maxHeight))
    return dst


def check_if_picture(colored_img, greyscale_img, mask):
    picture = colored_img[mask == 255]
    hist = compute_histogram(picture)
    
    # -- CHECK IF WHITE STATUE
    n = (hist[:256] + hist[256:512] + hist[512:768])
    print('Sum n > 200', np.sum(n[STATUE_GRAY_HIST_THRESH:]))
    print('Sum of n < 200', np.sum(n[:STATUE_GRAY_HIST_THRESH]))
    if np.sum(n[STATUE_GRAY_HIST_THRESH:]) > np.sum(n[:STATUE_GRAY_HIST_THRESH]):
        print("Not picture -> STATUE")
        return False
    
    ent = entropy(hist)
    if ent <= ENTROPY_SURE_NOT_PICTURE_THRESH:
        if DEBUG:
            print('Component is not a picture. Cause: ENTROPY')
        return False
    elif ent >= ENTROPY_SURE_PICTURE_THRESH:
        return True
    else:
        gray_picture = greyscale_img[mask == 255]
        mean = gray_picture.mean()
        if mean < PICTURE_GRAY_THRESH:
            return True
        else:
            if DEBUG:
                print('Component is not a picture. Cause: GRAY')
            return False


def create_non_repeated_couples_of_indexes(n_indexes):
    idxs = np.arange(n_indexes)
    idxs = np.vstack((np.repeat(idxs, n_indexes), np.tile(idxs, n_indexes))).T
    idxs = idxs[idxs[:, 0] != idxs[:, 1]]
    idxs = np.sort(idxs, axis=1)
    idxs = np.unique(idxs, axis=0)
    return idxs


def find_intersection(l1_start, l1_end, l2_start, l2_end):
    assert len(l1_start) == 2, 'len(l1_start) must be equal to 2'
    assert len(l1_end) == 2, 'len(l1_end) must be equal to 2'
    assert len(l2_start) == 2, 'len(l2_start) must be equal to 2'
    assert len(l2_end) == 2, 'len(l2_end) must be equal to 2'
    x1, y1 = l1_start
    x2, y2 = l2_start
    x3, y3 = l1_end
    x4, y4 = l2_end

    if x3 == x1:
        x3 = x3 + 0.01
    if x4 == x2:
        x4 = x4 + 0.01
    a = (y3 - y1) / (x3 - x1)
    b = (y4 - y2) / (x4 - x2)
    if a == b:
        a = (0.01 + y3 - y1) / (x3 - x1)
    x = (x1 * a - x2 * b - y1 + y2) / (a - b)
    y = (x - x1) * a + y1
    #return np.array([int(x), int(y)], dtype=np.int)
    return np.round([x, y]).astype(np.int)


def find_midpoint(p1, p2):
    assert len(p1) == 2, 'len(p1) must be equal to 2'
    assert len(p2) == 2, 'len(p2) must be equal to 2'
    return np.round([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]).astype(np.int)
    #return np.array([int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)], dtype=np.int)


def find_biggest_halfrectangle(p1, p2, p3, p4):
    assert len(p1) == 2, 'len(p1) must be equal to 2'
    assert len(p2) == 2, 'len(p2) must be equal to 2'
    assert len(p3) == 2, 'len(p3) must be equal to 2'
    assert len(p4) == 2, 'len(p4) must be equal to 2'
    c = find_intersection(p1, p3, p2, p4)
    v_inf = find_intersection(p1, p4, p2, p3)
    h_inf = find_intersection(p1, p2, p3, p4)
    m12 = find_intersection(p1, p2, c, v_inf)
    m23 = find_intersection(p2, p3, c, h_inf)
    m34 = find_intersection(p3, p4, c, v_inf)
    m41 = find_intersection(p4, p1, c, h_inf)
    bottom_side = False
    left_side = False
    if find_distance_squared(c, m12) < find_distance_squared(c, m34):
        bottom_side = True
    if find_distance_squared(c, m23) < find_distance_squared(c, m41):
        left_side = True
    if bottom_side and left_side:
        return np.array([m41, c, m34, p4], dtype=np.int), RECTANGLE_SECTOR_BL
    elif bottom_side and not left_side:
        return np.array([c, m23, p3, m34], dtype=np.int), RECTANGLE_SECTOR_BR
    elif not bottom_side and left_side:
        return np.array([p1, m12, c, m41], dtype=np.int), RECTANGLE_SECTOR_TL
    else: # not bottom_side and not left_side
        return np.array([m12, p2, m23, c], dtype=np.int), RECTANGLE_SECTOR_TR


def rectify_image(img, corners):
    assert corners.shape == (4, 2), 'corners.shape must be (4, 2)'
    corners = corners.astype(np.float32)
    new_vertices = corners.copy()
    for i in range(RECTIFY_IMAGE_ITER):
        new_vertices, sector = find_biggest_halfrectangle(new_vertices[0], new_vertices[1],
                                                          new_vertices[2], new_vertices[3])

    w = int((2 ** RECTIFY_IMAGE_ITER) * max(find_distance(new_vertices[0], new_vertices[1]),
                                            find_distance(new_vertices[2], new_vertices[3])))
    h = int((2 ** RECTIFY_IMAGE_ITER) * max(find_distance(new_vertices[1], new_vertices[2]),
                                            find_distance(new_vertices[3], new_vertices[0])))

    new_vertices = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    #print("NEW rectify w/h: ", w/h)
    mat = cv2.getPerspectiveTransform(corners, new_vertices)
    return cv2.warpPerspective(img, mat, (w, h))
