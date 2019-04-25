from lines import *

import numpy as np
import cv2
from sklearn.cluster import KMeans


class Component:

    def __init__(self, label, components_mask, stats, flag):
        self.label = label
        self.mask = np.zeros_like(components_mask, dtype=np.uint8)
        self.mask[components_mask == label] = 255
        self.stats = stats.reshape(-1)
        self.picture_part_flag = flag

    @property
    def height(self):
        return self.stats[cv2.CC_STAT_HEIGHT]

    @property
    def width(self):
        return self.stats[cv2.CC_STAT_WIDTH]

    @property
    def component_start_x(self):
        return self.stats[cv2.CC_STAT_LEFT]

    @property
    def component_start_y(self):
        return self.stats[cv2.CC_STAT_TOP]

    @property
    def area(self):
        return self.stats[cv2.CC_STAT_AREA]

    def get_hough_lines(self):
        h = self.height
        w = self.width
        min_ = np.min([h, w])
        th = np.int(np.round(min_ * 0.3))
        canny = cv2.Canny(self.mask, 50, 150)
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180., th, minLineLength=min_*0.05)
        if lines is None or len(lines) == 0:
            return None
        lines = lines.reshape((-1, 2, 2))
        out = np.zeros_like(self.mask)
        for line in lines:
            out = cv2.line(out, tuple(line[0]), tuple(line[1]), 255)
        return lines

    def search_for_optimal_lines(self, lines):
        if len(lines) < 4:
            return None
        equations = to_cartesian_equation(lines)

        cosines = np.cos(np.arctan(equations[:, 0]))
        clusters = KMeans(2).fit_predict(cosines.reshape((-1, 1)))
        labels = np.unique(clusters)

        optimal_lines = []
        for label in labels:
            current_equations = equations[clusters == label]
            current_lines = lines[clusters == label]
            current_lines_x = current_lines[:, :, 0]
            mean_x = np.mean(np.mean(current_lines_x, axis=1))
            n_lines = len(current_equations)
            if n_lines == 1:
                optimal_lines.append((current_lines[0], None, None))
                continue
            idxs = np.arange(n_lines)
            idxs = np.vstack((np.repeat(idxs, n_lines), np.tile(idxs, n_lines))).T
            idxs = idxs[idxs[:, 0] != idxs[:, 1]]
            idxs = np.sort(idxs, axis=1)
            idxs = np.unique(idxs, axis=0)
            couples = np.empty(shape=(len(idxs), 2, 2), dtype=np.float)
            couples[:, 0] = current_equations[idxs[:, 0]]
            couples[:, 1] = current_equations[idxs[:, 1]]
            points = np.empty_like(couples)
            points[:, 0] = compute_in_point(couples[:, 0], mean_x)
            points[:, 1] = compute_in_point(couples[:, 1], mean_x)
            distances = np.sqrt(np.sum(np.square(np.diff(points, axis=1)), axis=2))
            max_ = np.argmax(distances)
            max_idx = idxs[max_]
            optimal_lines.append((current_lines[max_idx[0]], current_lines[max_idx[1]], distances[max_]))
        out = np.zeros_like(self.mask)
        for l in optimal_lines:
            l1 = l[0]
            l2 = l[1]
            out = cv2.line(out, tuple(l1[0]), tuple(l1[1]), 255)
            if l2 is not None:
                out = cv2.line(out, tuple(l2[0]), tuple(l2[1]), 255)
        return optimal_lines

    def get_vertices(self, img):
        h, w = img.shape[:2]
        lines = self.get_hough_lines()
        if lines is None:
            return None, None

        optimal_lines = self.search_for_optimal_lines(lines)
        if optimal_lines is None or len(optimal_lines) < 2:
            return None, None
        th = np.min([self.height, self.width]) * 0.5
        out_lines = []
        for line1, line2, distance in optimal_lines:
            if line2 is None:
                out_lines.append(np.array([line1]))
            elif distance < th:
                out_lines.append(np.array([line1]))
            else:
                out_lines.append(np.array([line1, line2]))
        image_intersections = []
        real_intersections = []
        for line1 in out_lines[0]:
            for line2 in out_lines[1]:
                line1 = line1.reshape((-1, 2, 2))
                line2 = line2.reshape((-1, 2, 2))
                intersection = lines_intersection(line1, line2)
                real_intersections.append([intersection[0, 0], intersection[0, 1]])
                intersection = np.round(intersection).astype(np.int)
                intersection[0, 0] = np.clip(intersection[0, 0], 0, w)
                intersection[0, 1] = np.clip(intersection[0, 1], 0, h)
                image_intersections.append([intersection[0, 0], intersection[0, 1]])
        image_intersections = np.array(image_intersections, dtype=np.int)
        real_intersections = np.array(real_intersections, dtype=np.float)
        return image_intersections, real_intersections

    def check_if_contained_in_another_component(self, global_mask):
        x = self.component_start_x
        y = self.component_start_y
        area = self.height * self.width
        m = global_mask[y:y + self.height, x:x + self.width]
        s = np.sum(m == 255)
        if s == area:
            return True, global_mask
        else:
            return False, global_mask
