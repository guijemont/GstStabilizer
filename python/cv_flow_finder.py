#!/usr/bin/env python

from itertools import izip
import random

import cv, cv2

import numpy

from cv_gst_util import *

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing

class Finder(object):
    def __init__(self, *args, **kw):
        super(Finder, self).__init__(*args, **kw)

    def optical_flow(self, buf0, buf1):
        """
        Return two sets of coordinates (c0, c1), in buf0 and buf1 respectively,
        such that c1[i] is the position in buf1 of the feature that is at c0[i]
        in buf0.
        """
        if buf0 is None:
            return None

        img0 = img_of_buf(buf0)
        img1 = img_of_buf(buf1)

        return self.optical_flow_img(img0, img1)


class LucasKanadeFinder(Finder):
    def __init__(self, corner_count,
                       corner_quality_level,
                       corner_min_distance,
                       win_size,
                       pyramid_level,
                       max_iterations,
                       epsilon,
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

    def optical_flow_img(self, img0, img1):
        corners0 = self._features(img0)

        n_features = len(corners0)

        corners1, status, track_errors = cv.CalcOpticalFlowPyrLK (
                     img0, img1, None, None,
                     corners0,
                     (self.win_size,) * 2, # win size
                     self.pyramid_level, # pyramid level
                     (cv.CV_TERMCRIT_ITER|cv.CV_TERMCRIT_EPS, # stop type
                      self.max_iterations, # max iterations
                      self.epsilon), # min accuracy
                     0) # flags

        corners0 = self._filter_features(corners0, status)
        corners1 = self._filter_features(corners1, status)
        track_errors = self._filter_features(track_errors, status)

        print "%d features found, %d matched"  % (n_features, len(corners0)), ';',
        print "errors min/max/avg:", (min(track_errors),
                                      max(track_errors),
                                      sum(track_errors)/len(track_errors))

        return (corners0, corners1)

    def _features(self, img):
        img_size = cv.GetSize(img)
        eigImage = cv.CreateImage(img_size, cv.IPL_DEPTH_8U, 1)
        tempImage = cv.CreateImage(img_size, cv.IPL_DEPTH_8U, 1)

        mask = None
        features = cv.GoodFeaturesToTrack(img, eigImage, tempImage,
                                          self.corner_count, #number of corners to detect
                                          self.corner_quality_level, #Multiplier for the max/min
                                                #eigenvalue; specifies the minimal
                                                #accepted quality of image corners
                                          self.corner_min_distance, # minimum distance between returned corners
                                          mask=self.mask
                                          )

        return cv.FindCornerSubPix(img, features, (10, 10), (-1, -1),
                                   (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS,
                                    20, 0.03))

    def _filter_features(self, features, flter):
        #filtered_errors = [err for (err, status) in zip(errors, flter) if status]
        #n = len(filtered_errors) / 2 # we want to keep the best third only
        #admissible_error = sorted(filtered_errors)[n]
        return [ p for (keep,p) in izip(flter, features) if keep]


class HornSchunckFinder(Finder):
    def __init__(self, resize_ratio, *args, **kw):
        super(HornSchunckFinder, self).__init__(*args, **kw)

        self.resize_ratio = resize_ratio

    def optical_flow_img(self, img0, img1):
        assert(img0.width % self.resize_ratio == 0)
        assert(img0.height % self.resize_ratio == 0)

        width = img0.width / self.resize_ratio
        height = img0.height / self.resize_ratio

        small_img0 = resize(img0, width, height)
        small_img1 = resize(img1, width, height)

        velx = cv.CreateMat(height, width, cv.CV_32FC1)
        vely = cv.CreateMat(height, width, cv.CV_32FC1)

        cv.CalcOpticalFlowHS(small_img0, small_img1, False, velx, vely,
                             1.0, # lagrangian multiplier (wtf?)
                             (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS,
                              10, 0.01))

        origin, dest = self._velocities_to_vectors(velx, vely, width, height)

        converted_origin = self._convert_points(origin)
        converted_dest = self._convert_points(dest)

        return converted_origin, converted_dest

    def _velocities_to_vectors(self, velx, vely, width, height):
        origin = []
        dest = []
        # we take only one vector in 100, because we don't need that crazy
        # density
        # FIXME: make this configurable
        for y in xrange(0,height,10):
            for x in xrange(0,width,10):
                origin.append((x, y))
                dest.append((x + velx[y,x], y + vely[y,x]))

        return origin, dest

    def _convert_points(self, vectors):
        l = []
        for (x, y) in vectors:
            # we add resize_ratio/2 so that the converted point is in (or near)
            # the middle of the area it represents in the big image, not at its
            # top left.
            l.append((x * self.resize_ratio + self.resize_ratio / 2,
                      y * self.resize_ratio + self.resize_ratio / 2))
        return l

class SURFFinder(Finder):
    def __init__(self, *args, **kw):
        super(SURFFinder, self).__init__(*args, **kw)
        self._mem_storage = cv.CreateMemStorage()
        self._surf = cv2.SURF(1000)

    def get_surf(self, img):
        # returns  (keypoints, descriptors) where:
        # - keypoints is a list of ((x, y), laplacian, size, direction, hessian)
        # - descriptor is a cvSeq (kinda list?) of lists of 128 floats each
        #return self._surf.detect(img, None)
        if type(img) == numpy.ndarray:
            if len(img.shape)==3 and img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv.fromarray(img)
        return cv.ExtractSURF(img, None, self._mem_storage, (1, 3000, 3, 4))

    def optical_flow_img(self, img0, img1):
        surf_keypoints0, surf_keypoints1, dists = self.matching_surf_keypoints(img0, img1)

        print "found %d matches" % len(surf_keypoints0)

        def surf_to_normal_point_list(surf_kp_list):
            return [p[0] for p in surf_kp_list]

        surf_keypoints0, surf_keypoints1 = self._filter_diverging_angles(surf_keypoints0,
                                                                         surf_keypoints1)

        keypoints0 = surf_to_normal_point_list(surf_keypoints0)
        #keypoints0 = self._refine_points(img0, keypoints0)
        keypoints1 = surf_to_normal_point_list(surf_keypoints1)
        #keypoints1 = self._refine_points(img1, keypoints1)

        return keypoints0, keypoints1

    def matching_surf_keypoints(self, img0, img1):
        # return a pair of list of SURF keypoints that are supposed to match
        # each other. SURF keypoints are in a tuple of the following format:
        # ((x, y), laplacian, size, dir, hessian)

        keypoints0, descriptors0 = self.get_surf(img0)
        print "img0: found %d points" % len(keypoints0)
        keypoints1, descriptors1 = self.get_surf(img1)
        print "img1: found %d points" % len(keypoints1)

        indices, dists = self._find_neighbours(descriptors0, descriptors1)

        (result_keypoints0, result_keypoints1) = ([], [])
        for idx0, idx1 in indices:
            result_keypoints0.append(keypoints0[idx0])
            result_keypoints1.append(keypoints1[idx1])

        return result_keypoints0, result_keypoints1, dists

    def _filter_diverging_angles(self, surf_keypoints0, surf_keypoints1):
        rotation_angles = []
        for spt0, spt1 in izip(surf_keypoints0, surf_keypoints1):
            dir0 = spt0[3]
            dir1 = spt1[3]
            angle = (dir1 - dir0) % 360

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

    def _refine_points(self, img, points):
        if isinstance(img, cv.iplimage):
            return cv.FindCornerSubPix(img, points, (10, 10), (-1, -1),
                (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS,
                 20, 0.03))
        else:
            points = numpy.asarray(points, dtype=numpy.float32)
            return cv2.cornerSubPix(img, points, (10, 10), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                     20, 0.03))
    def _find_neighbours(self, descriptors0, descriptors1):
        # return a list of pairs (idx0, idx1) such that descriptors0[idx0] is
        # very likely to describe the same feature as descriptors1[idx1]
        # using the same params as the find_obj.cpp demo from opencv

        haystack = numpy.asarray(descriptors0, dtype=numpy.float32)
        needles = numpy.asarray(descriptors1, dtype=numpy.float32)

        # FIXME: have that an instance member when we match against first pic
        flann = cv2.flann_Index(haystack,
                                {'algorithm': FLANN_INDEX_KDTREE,
                                'trees': 4})
        indices, dists = flann.knnSearch(needles, 2, params={})
        # descriptors0[indices[i][0]] <-> descriptors1[i]

        result = []
        result_dists = []

        for i, flann_idx, (small_dist, big_dist) in izip(xrange(len(needles)),
                                                         indices,
                                                         dists):
            if small_dist < 0.1 and small_dist < big_dist * 0.6:
                result.append((flann_idx[0], i))
                result_dists.append(small_dist)
        return result, result_dists


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
    demo = FinderDemo(SURFFinder(), *sys.argv[1:])
    demo.demo()
