import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max

from . import grasp as base_grasp


def _gr_text_to_no(l, offset=(0, 0)):
    """Transform a single point from a text line to an integer [y, x] pair."""
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


def _normalise_grasp_points(points):
    """Return points ordered around the rectangle with the longest edge first."""
    pts = np.array(points, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError("Grasp rectangles must contain four corner points.")

    center = pts.mean(axis=0)

    # Order corners by angle around the center to follow the polygon loop.
    angles = np.arctan2(pts[:, 0] - center[0], pts[:, 1] - center[1])
    order = np.argsort(angles)
    ordered = pts[order]

    # Find the edge with the greatest length and rotate so it starts at index 0.
    diffs = ordered[(np.arange(4) + 1) % 4] - ordered
    lengths = np.linalg.norm(diffs, axis=1)
    start_idx = int(np.argmax(lengths))
    aligned = np.roll(ordered, -start_idx, axis=0)

    return np.round(aligned).astype(np.int64)


class GraspRectangles(base_grasp.GraspRectangles):
    """Convenience container for loading and operating on sets of GraspRectangles."""

    def __init__(self, grs=None):
        super().__init__(grs=grs)

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return iter(self.grs)

    def __getattr__(self, attr):
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: [getattr(gr, attr)(*args, **kwargs) for gr in self.grs]
        return super().__getattr__(attr)

    @classmethod
    def load_from_array(cls, arr):
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
        grs = []
        with open(fname) as f:
            while True:
                p0 = f.readline()
                if not p0:
                    break
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    rect = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])
                    grs.append(GraspRectangle(rect))
                except ValueError:
                    continue
        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                grs.append(Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    def append(self, gr):
        self.grs.append(gr)

    def copy(self):
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        pos_out = np.zeros(shape) if position else None
        ang_out = np.zeros(shape) if angle else None
        width_out = np.zeros(shape) if width else None

        for gr in self.grs:
            rr, cc = gr.compact_polygon_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = gr.angle
            if width:
                width_out[rr, cc] = gr.length

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        a = np.stack([gr.points for gr in self.grs])
        if pad_to and pad_to > len(self.grs):
            a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int64)

    @property
    def center(self):
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(np.int64)


class GraspRectangle(base_grasp.GraspRectangle):
    """Representation of a grasp rectangle with normalised point ordering."""

    def __init__(self, points, normalise=True):
        if normalise:
            points = _normalise_grasp_points(points)
        super().__init__(np.array(points, dtype=np.int64))

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2

    @property
    def as_grasp(self):
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        return self.points.mean(axis=0).astype(np.int64)

    @property
    def length(self):
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        return Grasp(self.center, self.angle, self.length / 3, self.width).as_gr.polygon_coords(shape)

    def iou(self, gr, angle_threshold=np.pi / 6):
        if abs((self.angle - gr.angle + np.pi / 2) % np.pi - np.pi / 2) > angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except ValueError:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union

    def copy(self):
        return GraspRectangle(self.points.copy(), normalise=False)

    def offset(self, offset):
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int64)

    def scale(self, factor):
        if factor != 1.0:
            self.points = (self.points * factor).astype(np.int64)

    def plot(self, ax, color=None):
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        T = np.array(
            [
                [1 / factor, 0],
                [0, 1 / factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int64)


class Grasp:
    """A grasp represented by center, rotation angle, and gripper width/length."""

    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = angle
        self.length = length
        self.width = width

    @property
    def as_gr(self):
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        rect = np.array(
            [
                [y1 - self.width / 2 * xo, x1 - self.width / 2 * yo],
                [y2 - self.width / 2 * xo, x2 - self.width / 2 * yo],
                [y2 + self.width / 2 * xo, x2 + self.width / 2 * yo],
                [y1 + self.width / 2 * xo, x1 + self.width / 2 * yo],
            ]
        )
        return GraspRectangle(rect, normalise=False)

    def max_iou(self, grs, angle_threshold=None):
        self_gr = self.as_gr
        max_iou = 0
        for gr in grs:
            angle_thresh_rad = np.deg2rad(angle_threshold) if angle_threshold is not None else None
            iou = self_gr.iou(gr, angle_thresh_rad) if angle_thresh_rad is not None else self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        self.as_gr.plot(ax, color)

    def to_jacquard(self, scale=1):
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (
            self.center[1] * scale,
            self.center[0] * scale,
            -1 * self.angle * 180 / np.pi,
            self.length * scale,
            self.width * scale
        )


def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2

        grasps.append(g)

    return grasps
