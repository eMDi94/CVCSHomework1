from utils import *
from htrdc import HTRDC, undistort
from components import Component
from parameters import *

import numpy as np
import cv2


def resize_when_too_big(img, threshold_w_h):
    h = int(img.shape[0])
    w = int(img.shape[1])
    thr_w, thr_h = threshold_w_h
    if h > thr_h or w > thr_h:
        h_ratio = thr_h / h
        w_ratio = thr_w / w
        ratio = min(h_ratio, w_ratio)
        img = resize_to_ratio(img, ratio)
    return img


def read_undistorted_image_color_grayscale(img_file):
    img = cv2.imread(img_file)
    img = resize_when_too_big(img, PICTURE_SIZE_THRESH_W_H)
    gray = convert_to(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLURRING_GAUSSIAN_KERNEL_SIZE, BLURRING_GAUSSIAN_SIGMA)
    edges = cv2.Canny(gray, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    k = HTRDC(edges, (HTRDC_K_START, HTRDC_K_END), HTRDC_N, HTRDC_EPSILON)
    img = undistort(img, k)
    gray = convert_to(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def erode_dilate(img):
    img = cv2.erode(img, np.ones((3, 3), dtype=np.uint8))
    img = cv2.dilate(img, np.ones(DILATE_KERNEL_SIZE, dtype=np.uint8), iterations=DILATE_ITERATIONS)
    img = cv2.erode(img, np.ones(EROSION_KERNEL_SIZE, dtype=np.uint8), iterations=EROSION_ITERATIONS)
    return img


def draw_border_for_picture_parts(drawing):
    flag = False

    sm_column = np.sum(drawing, axis=0)
    if sm_column[0] > 0:
        drawing[:, :5] = 0
        flag = True
    if sm_column[-1] > 0:
        drawing[:, -5:] = 0
        flag = True

    sm_row = np.sum(drawing, axis=1)
    if sm_row[0] > 0:
        drawing[:5, :] = 0
        flag = True
    if sm_row[-1] > 0:
        drawing[-5:, :] = 0
        flag = True

    return drawing, flag


def connected_components_segmentation(img):
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                ADAPTIVE_THRESHOLD_KERNEL_SIZE, ADAPTIVE_THRESHOLD_C)
    img = cv2.medianBlur(img, 3)
    img = erode_dilate(img)

    _, labeled_img = cv2.connectedComponentsWithAlgorithm(img, 8, cv2.CV_32S, cv2.CCL_GRANA)
    labels = np.unique(labeled_img)
    labels = labels[labels != 0]

    components = []
    for label in labels:
        mask = np.zeros_like(labeled_img, dtype=np.uint8)
        mask[labeled_img == label] = 255

        # Compute the convex hull
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = []
        for cnt in contours:
            hull.append(cv2.convexHull(cnt, False))
        drawing = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(len(contours)):
            drawing = cv2.drawContours(drawing, hull, i, 255, -1, 8)

        single_component, flag = draw_border_for_picture_parts(drawing)

        _, connected_component, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(single_component, 8,
                                                                                          cv2.CV_32S, cv2.CCL_GRANA)
        valid_labels = np.argwhere(stats[:, cv2.CC_STAT_AREA] >= LABEL_AREA_THRESHOLD)
        if valid_labels[0] == 0:
            valid_labels = valid_labels[1:]
        for valid_label in valid_labels:
            component = Component(valid_label, connected_component, stats[valid_label], flag)
            components.append(component)
    components.sort(key=lambda x: x.area, reverse=True)
    return components, img


def show_vertices(img, image_vertices):
    img_c = img.copy()
    for i in range(len(image_vertices)):
        vertex = tuple(image_vertices[i])
        img_c = cv2.circle(img_c, vertex, 4, [0, 255, 0], thickness=-1)
    show(img_c)


def show_rectangle(img, sorted_vertices):
    img_lines = img.copy()
    cv2.line(img_lines , tuple(sorted_vertices[0,:]), tuple(sorted_vertices[1,:]), (0, 255, 0), 3)
    cv2.line(img_lines , tuple(sorted_vertices[1,:]), tuple(sorted_vertices[2,:]), (0, 255, 0), 3)
    cv2.line(img_lines , tuple(sorted_vertices[2,:]), tuple(sorted_vertices[3,:]), (0, 255, 0), 3)
    cv2.line(img_lines , tuple(sorted_vertices[3,:]), tuple(sorted_vertices[0,:]), (0, 255, 0), 3)
    show(img_lines)


def main():
    img, gray = read_undistorted_image_color_grayscale('./InputImages/110.jpg')
    gray = cv2.GaussianBlur(gray, BLURRING_GAUSSIAN_KERNEL_SIZE, BLURRING_GAUSSIAN_SIGMA)
    components, gray = connected_components_segmentation(gray)
    global_mask = np.zeros_like(gray, dtype=np.uint8)

    for component in components:
        is_contained, global_mask = component.check_if_contained_in_another_component(global_mask)

        if is_contained is True:
            continue
        if check_if_picture(img, gray, component.mask) is False:
            continue
        else:
            global_mask[component.mask == 255] = 255

        image_vertices, real_vertices = component.get_vertices(gray)
        if image_vertices is None:
            continue

        if DEBUG is True:
            show_vertices(img, image_vertices)

        if len(image_vertices) == 4:
            sorted_vertices = sort_corners(image_vertices)
            if DEBUG is True:
                show_rectangle(img, sorted_vertices)

            final = rectify_image(img, sorted_vertices)
            if final is not None and component.picture_part_flag is False:
                show(final)


if __name__ == '__main__':
    main()
