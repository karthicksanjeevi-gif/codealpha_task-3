"""
SORT: Simple Online and Realtime Tracking
A simple, effective multi-object tracking algorithm.
Based on: https://github.com/abewley/sort
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def iou(bb_test, bb_gt):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    bb format: [x1, y1, x2, y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union = area_test + area_gt - intersection

    return intersection / union if union > 0 else 0.0


def convert_bbox_to_z(bbox):
    """Convert [x1,y1,x2,y2] → [cx, cy, s, r] (center, scale, aspect ratio)."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h       # scale (area)
    r = w / float(h) if h != 0 else 1.0
    return np.array([cx, cy, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """Convert Kalman state [cx, cy, s, r, ...] → [x1, y1, x2, y2]."""
    w = np.sqrt(abs(x[2] * x[3]))
    h = x[2] / w if w != 0 else 0
    box = [x[0] - w / 2.0, x[1] - h / 2.0,
           x[0] + w / 2.0, x[1] + h / 2.0]
    if score is None:
        return np.array(box).reshape((1, 4))
    return np.array([box[0], box[1], box[2], box[3], score]).reshape((1, 5))


class KalmanBoxTracker:
    """
    Represents a single tracked object using a Kalman filter.
    State vector: [cx, cy, s, r, vx, vy, vs]
    """
    count = 0  # class-level counter for unique IDs

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix (we observe cx, cy, s, r)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        self.kf.R[2:, 2:] *= 10.0   # measurement noise
        self.kf.P[4:, 4:] *= 1000.0 # high uncertainty for unobserved velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Hungarian algorithm to match detections to existing trackers.
    Returns: matches, unmatched_detections, unmatched_trackers
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0, 5), dtype=int))

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=float)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)

    unmatched_detections = [d for d in range(len(detections))
                            if d not in matched_indices[:, 0]]
    unmatched_trackers  = [t for t in range(len(trackers))
                           if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """
    SORT multi-object tracker.

    Parameters
    ----------
    max_age       : frames to keep a tracker alive without a matching detection
    min_hits      : min detections before a tracker is shown
    iou_threshold : IoU threshold for matching
    """

    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Update tracker state.

        Parameters
        ----------
        dets : np.ndarray, shape (N, 5)  — [x1, y1, x2, y2, score]

        Returns
        -------
        np.ndarray, shape (M, 5) — [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # Predict new locations of existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4], trks[:, :4], self.iou_threshold
        )

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        # Collect results and remove dead trackers
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update <= self.max_age and
                    (trk.hit_streak >= self.min_hits or
                     self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))

        # Remove stale trackers
        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.max_age]

        if ret:
            return np.concatenate(ret)
        return np.empty((0, 5))
