#!/usr/bin/env python
#
# Copyright 2011 Igalia S.L. and Guillaume Emont
# Contact: Guilaume Emont <guijemont@igalia.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import gobject,gst

import cPickle

from cv_gst_util import *

from cv_flow_finder import LucasKanadeFinder, SURFFinder


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

    # Algorithms to chose from:
    LUCAS_KANADE = 1
    SURF = 2

    corner_count = gobject.property(type=int,
                                 default=50,
                                 blurb='number of corners to detect')
    corner_quality_level = gobject.property(type=float,
                                            default=0.1,
                                            blurb='Multiplier for the max/min eigenvalue; specifies the minimal accepted quality of image corners')
    corner_min_distance = gobject.property(type=int,
                                           default=50,
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
    algorithm = gobject.property(type=int,
                                 default=LUCAS_KANADE,
                                 blurb= """algorithm to use:
                                 %d: Lucas Kanade (discreet, fast, precise, not good for big changes between frames)
                                 %d: SURF (Speeded Up Robust Feature, finds features, finds them again)""" % (LUCAS_KANADE, SURF))


    def __init__(self, *args, **kw):
        super(OpticalFlowFinder, self).__init__(*args, **kw)

        self.srcpad = gst.Pad(self.src_template)
        self.add_pad(self.srcpad)

        self.sinkpad = gst.Pad(self.sink_template)
        self.sinkpad.set_chain_function(self._chain)
        self.add_pad(self.sinkpad)

        self._previous_img = None
        self._previous_blob = None

        self._finder = None

    def _chain(self, pad, buf):
        img = img_of_buf(buf)

        if self._previous_img is not None:
            flow, blob = self._finder.optical_flow_img(self._previous_img,
                                                       img,
                                                       self._previous_blob)
        else:
            flow, blob = None, None
        self._previous_img = img
        self._previous_blob = blob

        pickled_flow = cPickle.dumps(flow, self.PICKLE_FORMAT)
        new_buf = gst.Buffer(pickled_flow)
        new_buf.stamp(buf)

        return self.srcpad.push(new_buf)

    def do_change_state(self, state_change):
        if state_change == gst.STATE_CHANGE_NULL_TO_READY:
            self._finder = self._create_finder()
        elif state_change == gst.STATE_CHANGE_READY_TO_NULL:
            self._finder = None
            self._previous_img = None
            self._previous_blob = None

        return gst.Element.do_change_state(self, state_change)

    def _create_finder(self):
        if self.algorithm == self.LUCAS_KANADE:
            finder = LucasKanadeFinder(self.corner_count,
                                             self.corner_quality_level,
                                             self.corner_min_distance,
                                             self.win_size,
                                             self.pyramid_level,
                                             self.max_iterations,
                                             self.epsilon)
        elif self.algorithm == self.SURF:
            finder = SURFFinder()
        else:
            raise ValueError("Unknown algorithm")
        return finder


gobject.type_register (OpticalFlowFinder)
ret = gst.element_register (OpticalFlowFinder, 'opticalflowfinder')

