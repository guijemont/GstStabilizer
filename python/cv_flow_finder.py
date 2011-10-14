#!/usr/bin/env python

from itertools import izip

import cv

from cv_gst_util import *

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
