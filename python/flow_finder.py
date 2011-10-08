#!/usr/bin/env python

import gobject,gst

import array, cPickle
from itertools import izip

import cv

from cv_gst_util import *

import cv_flow_finder


class OpticalFlowFinder(gst.Element):
    __gstdetails__ = ("Optical flow finder",
                    "Filter/Video",
                    "find optical flow between frames",
                    "Guillaume Emont")
    sink_template = gst.PadTemplate ("sink",
                                   gst.PAD_SINK,
                                   gst.PAD_ALWAYS,
                                   gst.Caps('video/x-raw-gray,depth=8'))

    src_template = gst.PadTemplate ("source",
                                     gst.PAD_SRC,
                                     gst.PAD_ALWAYS,
                                     gst.Caps('application/x-motion-flow'))

    __gsttemplates__ = (sink_template, src_template)

    PICKLE_FORMAT = 2

    corner_count = gobject.property(type=int,
                                 default=20,
                                 blurb='number of corners to detect')
    corner_quality_level = gobject.property(type=float,
                                            default=0.1,
                                            blurb='Multiplier for the max/min eigenvalue; specifies the minimal accepted quality of image corners')
    corner_min_distance = gobject.property(type=int,
                                           default=200,
                                           blurb='Limit, specifying the minimum possible distance between the detected corners; Euclidian distance is used')
    win_size = gobject.property(type=int,
                                default=30,
                                blurb='Size of the search window of each pyramid level')
    pyramid_level = gobject.property(type=int,
                                     default=4,
                                     blurb='Maximal pyramid level number. If 0 , pyramids are not used (single level), if 1 , two levels are used, etc')
    max_iterations = gobject.property(type=int,
                                      default=50,
                                      blurb='maximum number of iterations to calculate optical flow')
    epsilon = gobject.property(type=float,
                                    default=0.001,
                                    blurb='terminate when we reach that difference or smaller')

    ignore_box_min_x = gobject.property(type=int,
                                        default=-1,
                                        blurb='left limit of the ignore box, deactivated if -1')
    ignore_box_max_x = gobject.property(type=int,
                                        default=-1,
                                        blurb='right limit of the ignore box, deactivated if -1')
    ignore_box_min_y = gobject.property(type=int,
                                        default=-1,
                                        blurb='top limit of the ignore box, deactivated if -1')
    ignore_box_max_y = gobject.property(type=int,
                                        default=-1,
                                        blurb='top limit of the ignore box, deactivated if -1')

    def __init__(self):
        gst.Element.__init__(self)

        self.sinkpad = gst.Pad(self.sink_template)
        self.sinkpad.set_chain_function(self.chain)
        self.add_pad(self.sinkpad)

        self.srcpad = gst.Pad(self.src_template)
        self.add_pad(self.srcpad)

        self._previous_frame = None

        self._finder = cv_flow_finder.LucasKanadeFinder(self.corner_count,
                                                        self.corner_quality_level,
                                                        self.corner_min_distance,
                                                        self.win_size,
                                                        self.pyramid_level,
                                                        self.max_iterations,
                                                        self.epsilon)


    def chain(self, pad, buf):

        if self._finder.mask is None and self._has_ignore_box():
            # we consider the buffer height and width are constant, the whole
            # algorithm depends on it anyway. Shouldn't we enforce that somewhere?
            caps_struct = buf.get_caps()[0]
            height = caps_struct['height']
            width = caps_struct['width']
            self._finder.mask = cv.CreateMatHeader(height, width, cv.CV_8UC1)
            data = array.array('B', '\1' * width * height)
            for x in xrange(self.ignore_box_min_x, self.ignore_box_max_x + 1):
                for y in xrange(self.ignore_box_min_y, self.ignore_box_max_y + 1):
                    data[y*height + x] = 0
            cv.SetData(self._finder.mask, data.tostring())

        flow = self._finder.optical_flow(self._previous_frame, buf)
        if flow is None:
            filtered_flow = None
        else:
            filtered_flow = self._filter_flow(flow)

        pickled_flow = cPickle.dumps(filtered_flow, self.PICKLE_FORMAT)
        new_buf = gst.Buffer (pickled_flow)
        new_buf.stamp(buf)

        self._previous_frame = buf

        return self.srcpad.push(new_buf)


    def _filter_flow(self, (points0, points1)):
        #filtered_errors = [err for (err, status) in zip(errors, flter) if status]
        #n = len(filtered_errors) / 2 # we want to keep the best third only
        #admissible_error = sorted(filtered_errors)[n]
        #return [ p for (status,p, err) in zip(flter, features, errors)
        #            if status and not self._in_ignore_box(p)] # and err < admissible_error]

        if not self._has_ignore_box():
            return (points0, points1)

        result0 = []
        result1 = []

        for p0, p1 in izip(points0, points1):
            if not (self._in_ignore_box(p0) or self._in_ignore_box(p1)):
                result0.append(p0)
                result1.append(p1)

        return (result0, result1)


    def _has_ignore_box(self):
        return (-1) not in (self.ignore_box_min_x, self.ignore_box_max_x,
                            self.ignore_box_min_y, self.ignore_box_max_y)

    def _in_ignore_box(self, (x, y)):
        return self._has_ignore_box() and x >= self.ignore_box_min_x and x <= self.ignore_box_max_x and y >= self.ignore_box_min_y and y <= self.ignore_box_max_y



gobject.type_register (OpticalFlowFinder)
ret = gst.element_register (OpticalFlowFinder, 'opticalflowfinder')

