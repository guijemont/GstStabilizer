#!/usr/bin/env python

from itertools import izip
import random

import cv2

import numpy

from cv_gst_util import *

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing

class Finder(object):
    def __init__(self, *args, **kw):
        super(Finder, self).__init__(*args, **kw)

    def optical_flow(self, buf0, buf1, blob_buf0=None):
        """
        Returns the flow and the blob for buf1
        """
        if buf0 is None:
            return None,None

        img0 = img_of_buf(buf0)
        img1 = img_of_buf(buf1)
        return self.optical_flow_img(img0, img1, blob_buf0)

    def optical_flow_img(self, img0, img1, blob_buf0=None):
        raise NotImplementedError()

    def warp_blob(self, blob, transform_matrix):
        raise NotImplementedError()

class LucasKanadeFinder(Finder):
    def __init__(self, corner_count=50,
                       corner_quality_level=0.1,
                       corner_min_distance=50,
                       win_size=30,
                       pyramid_level=4,
                       max_iterations=50,
                       epsilon=0.001,
                       *args, **kw):

        super(LucasKanadeFinder, self).__init__(*args, **kw)

        self.corner_count = corner_count
        self.corner_quality_level = corner_quality_level
        self.corner_min_distance = corner_min_distance
        self.win_size = win_size
        self.pyramid_level = pyramid_level
        self.max_iterations = max_iterations
        self.epsilon = epsilon

        self.mask = None

    def optical_flow_img(self, img0, img1, blob_buf0=None):
        # for us, blob_buf0 is in the format:
        # corners
        # for now.
        # TODO: add pyramid?

        if blob_buf0 is not None and len(blob_buf0) > self.corner_count / 2:
            corners0 = blob_buf0
        else:
            corners0 = self._features(img0)

        n_features = len(corners0)

        corners1, status, errors = cv2.calcOpticalFlowPyrLK(
                    img0, img1, corners0, None,
                    winSize=(self.win_size,) * 2,
                    maxLevel=self.pyramid_level,
                    criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
                              self.max_iterations, self.epsilon)
                    )

        # these are a few workarounds because openCV return things in a format
        # slightly different from what we want.
        if status.dtype == numpy.uint8:
            status.dtype = numpy.bool8
        if len(status.shape) > 1:
            assert(status.shape[1] == 1)
            status.shape = status.shape[:1]
        if len(errors.shape) > 1:
            assert(errors.shape[1] == 1)
            errors.shape = errors.shape[:1]

        corners0 = corners0[status]
        corners1 = corners1[status]

        errors = errors[status]
        print "%d features found, %d matched"  % (n_features, len(corners0)), ';',
        if len(errors):
            print "errors min/max/avg:", (min(errors),
                                          max(errors),
                                          (sum(errors)/len(errors)))

        return ((corners0, corners1), corners1)

    def warp_blob(self, blob, transform_matrix):
        if transform_matrix.dtype != numpy.float32:
            new_transform = numpy.ndarray(transform_matrix.shape,
                                          numpy.float32)
            new_transform[:] = transform_matrix
            transform_matrix = new_transform
        _, invert_transform = cv2.invert(transform_matrix)
        shape = (blob.shape[0], 3)
        extended_blob = numpy.ndarray(shape, dtype=blob.dtype)
        extended_blob[...,:2] = blob
        extended_blob[...,2] = 1.
        warped_blob = invert_transform.dot(extended_blob.transpose())
        return warped_blob.transpose()[...,:2]

    def _features(self, img):
        features = cv2.goodFeaturesToTrack(img,
                                           self.corner_count,
                                           self.corner_quality_level,
                                           self.corner_min_distance,
                                           mask=self.mask)
        if len(features.shape) == 3:
            assert(features.shape[1:] == (1,2))
            features.shape = (features.shape[0],2)

        cv2.cornerSubPix(img, features, (10, 10), (-1, -1),
                         (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
                         20, 0.03))
        return features

class FrameSURFInfo(object):
    def __init__(self, keypoints, descriptors, flann, *args, **kw):
        super(FrameSURFInfo, self).__init__(*args, **kw)
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.flann = flann

class SURFFinder(Finder):
    def __init__(self, *args, **kw):
        super(SURFFinder, self).__init__(*args, **kw)
        self._surf = cv2.SURF(1000, _extended=True)

    def get_surf(self, img):
        # returns  (keypoints, descriptors) where:
        # - keypoints is a list of cv2.Keypoint instances
        # - descriptor is a numpy.ndarray such that descriptors[i] is a
        # 128-float array which is the SURF descriptor of keypoints[i]

        keypoints, descriptors = self._surf.detect(img, None, False)

        # descriptors might not be provided in the right shape, but it should
        # have the right number of elements to be converted.
        descriptors.shape = (len(keypoints), 128)
        return keypoints, descriptors

    def optical_flow_img(self, img0, img1, blob0):
        # blob0 is a FrameSURFInfo
        surf_keypoints0, surf_keypoints1, dists, new_blob = self.matching_surf_keypoints(img0, img1, blob0)

        print "found %d matches" % len(surf_keypoints0)

        def surf_to_normal_point_array(surf_kp_list):
            # probably room for optimisation here
            return numpy.asarray([p.pt for p in surf_kp_list])

        surf_keypoints0, surf_keypoints1 = self._filter_diverging_angles(surf_keypoints0,
                                                                         surf_keypoints1)

        keypoints0 = surf_to_normal_point_array(surf_keypoints0)
        keypoints1 = surf_to_normal_point_array(surf_keypoints1)


        return (keypoints0, keypoints1), new_blob

    def warp_blob(self, blob, transform_matrix):
        # FIXME: actually implement this
        return None

    def matching_surf_keypoints(self, img0, img1, blob0):
        # return a pair of list of SURF keypoints that are supposed to match
        # each other. 

        if blob0 is None:
            keypoints0, descriptors0 = self.get_surf(img0)
            print "img0: found %d points" % len(keypoints0)
            flann0 = None
        else:
            keypoints0 = blob0.keypoints
            descriptors0 = blob0.descriptors
            flann0 = blob0.flann
            print "img0: using %d points from blob" % len(keypoints0)
            print "flann from blob:", flann0
        keypoints1, descriptors1 = self.get_surf(img1)
        print "img1: found %d points" % len(keypoints1)

        indices, dists, flann1 = self._find_neighbours(descriptors0, descriptors1, flann0)

        (result_keypoints0, result_keypoints1) = ([], [])
        for idx0, idx1 in indices:
            result_keypoints0.append(keypoints0[idx0])
            result_keypoints1.append(keypoints1[idx1])

        new_blob = FrameSURFInfo(keypoints1, descriptors1, flann1)

        return result_keypoints0, result_keypoints1, dists, new_blob

    def _filter_diverging_angles(self, surf_keypoints0, surf_keypoints1):
        rotation_angles = []
        for spt0, spt1 in izip(surf_keypoints0, surf_keypoints1):
            angle = (spt1.angle - spt0.angle) % 360

            if angle > 180:
                angle -= 360

            rotation_angles.append(angle)

        median = numpy.median(rotation_angles)

        res0, res1 = [], []
        dropped = 0
        for angle, spt0, spt1 in izip(rotation_angles,
                                      surf_keypoints0, surf_keypoints1):
            if abs(angle-median) < 20:
                res0.append(spt0)
                res1.append(spt1)
            else:
                dropped += 1

        if dropped != 0:
            print "dropped %d points that were badly orientated" % dropped

        return res0, res1

    def _find_neighbours(self, descriptors0, descriptors1, flann0):
        # return a list of pairs (idx0, idx1) such that descriptors0[idx0] is
        # very likely to describe the same feature as descriptors1[idx1]
        # using the same params as the find_obj.cpp demo from opencv

        flann1 = None

        if flann0 is None:
            flann1 = cv2.flann_Index(descriptors1,
                                    {'algorithm': FLANN_INDEX_KDTREE,
                                    'trees': 4})
            indices, dists = flann1.knnSearch(descriptors0, 2, params={})
            needles_number = len(descriptors0)
            # we did a search of descriptors0 in descriptors1, and we got the
            # indices in descriptors1 where we can find the elements of
            # descriptors0, indices are for descriptors1
            # descriptors1[indices[i][0]] <-> descriptors0[i]
        else:
            print "searching using flann", flann0
            needles_number = len(descriptors1)
            indices, dists = flann0.knnSearch(descriptors1, 2, params={})
            # reverse case of above, indices are for descriptors0
            # descriptors0[indices[i][0]] <-> descriptors1[i]

        result = []
        result_dists = []

        for i, flann_idx, (small_dist, big_dist) in izip(xrange(needles_number),
                                                         indices,
                                                         dists):
            if small_dist < big_dist * 0.6:
                if flann0 is not None:
                    result.append((flann_idx[0], i))
                else:
                    result.append((i, flann_idx[0]))
                result_dists.append(small_dist)
        return result, result_dists, flann1


class FinderDemo(object):
    def __init__(self, finder, path0, path1, pathout, *args, **kw):
        super(FinderDemo, self).__init__(*args, **kw)
        self._finder = finder
        self._image0 = cv2.imread(path0)
        self._image1 = cv2.imread(path1)
        self._pathout = pathout

    def demo(self):
        points0, points1 = self._finder.optical_flow_img(self._image0, self._image1)
        h0, w0 = self._image0.shape[:2]
        h1, w1 = self._image1.shape[:2]

        demo_image = numpy.zeros((max(h0, h1), w0+w1, 3))
        demo_image[:h0, :w0] = self._image0
        demo_image[:h1, w0:w0+w1] = self._image1

        for (i, (x0, y0), (x1, y1)) in izip(xrange(len(points0)), numpy.int32(points0), numpy.int32(points1)):
            color = (random.randint(0, 255),
                     random.randint(0, 255),
                     random.randint(0, 255))

            cv2.line(demo_image, (x0, y0), (x1+w0, y1), color)
            cv2.putText(demo_image, str(i),
                        (int((x0+x1+w0)/2), int((y0+y1)/2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1., # font scale
                        color)

        cv2.imwrite(self._pathout, demo_image)


if __name__ == '__main__':
    import sys
    algorithm = sys.argv[1]
    if algorithm == 'HS':
        finder = HornSchunckFinder(period=40)
    elif algorithm == 'LK':
        finder = LucasKanadeFinder()
    elif algorithm == 'SURF':
        finder = SURFFinder()
    else:
        print "Uknown algorithm!"
        syntax()

    if finder:
        demo = FinderDemo(finder, *sys.argv[2:])
        demo.demo()
