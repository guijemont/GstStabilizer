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

import gst, gobject

from flow_muxer import OpticalFlowMuxer
from cv_gst_util import *


class OpticalFlowRevert(OpticalFlowMuxer):
    __gstdetails__ = ("Optical flow revert",
                    "Filter/Video",
                    "Revert the effect of the flow",
                    "Guillaume Emont")
    main_sink_template = gst.PadTemplate ("mainsink",
                                          gst.PAD_SINK,
                                          gst.PAD_ALWAYS,
                                          gst.Caps('video/x-raw-rgb,depth=24'))
    src_template = gst.PadTemplate("src",
                                    gst.PAD_SRC,
                                    gst.PAD_ALWAYS,
                                    gst.Caps('video/x-raw-rgb,depth=24'))
    __gsttemplates__ = (OpticalFlowMuxer.flow_sink_template,
                        main_sink_template,
                        src_template)

    demo_mode = gobject.property(type=bool,
                                 default=False,
                                 blurb="Output a mix of the unstabilised and stabilised streams")

    def __init__(self, *args, **kw):
        super(OpticalFlowRevert, self).__init__(*args, **kw)

        self.srcpad = gst.Pad(self.src_template)
        self.add_pad(self.srcpad)

        self._last_output_img = None
        self._reference_transform = numpy.asarray([[1., 0., 0.],
                                                   [0., 1., 0.],
                                                   [0., 0., 1.]],
                                                   dtype=numpy.float128)

    def mux(self, buf, flow):
        if flow is None:
            self._last_output_img = img_of_buf(buf)
            return self.srcpad.push(buf)

        origins, ends = flow

        transform, mask = cv2.findHomography(origins, ends,
                                             method=cv2.RANSAC,
                                             ransacReprojThreshold=3)

        # we accumulate the transformations, so that we apply a transformation
        # relative to the first frame
        self._reference_transform = transform.dot(self._reference_transform)

        img = img_of_buf(buf)
        new_img = self._last_output_img.copy()

        new_img = cv2.warpPerspective(img,
                                      numpy.asarray(self._reference_transform,
                                                    dtype=numpy.float64),
                                      (img.shape[1], img.shape[0]),
                                      dst=new_img,
                                      flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_TRANSPARENT)

        self._last_output_img = new_img
        if not self.demo_mode:
            new_buf = buf_of_img(new_img, bufmodel=buf)
            return self.srcpad.push(new_buf)
        else:
            demo_img = img.copy()
            width = img.shape[1]
            mid_width = width/2
            demo_img[:, mid_width:] = new_img[:, mid_width:]
            new_buf = buf_of_img(demo_img, bufmodel=buf)
            return self.srcpad.push(new_buf)


gobject.type_register (OpticalFlowRevert)
ret = gst.element_register (OpticalFlowRevert, 'opticalflowrevert')
