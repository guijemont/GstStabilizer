#!/usr/bin/env python
import gobject,gst

import math

import cv2

from cv_gst_util import *

from flow_muxer import OpticalFlowMuxer


class ArrowDrawer(object):
    alpha = math.pi / 6.
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)

    def __init__(self, width=2, *args, **kw):
        super(ArrowDrawer, self).__init__(*args, **kw)

        self._width = width

    def draw_arrows(self, img, origins, ends):
        for origin, end in zip(origins, ends):
            self.draw_arrow(img, origin, end)

    def draw_arrow(self, img, origin, end):
        color = (255, 0, 0)
        def int_pos((x,y)):
            return (int(x), int(y))
        origin = int_pos(origin)
        end = int_pos(end)
        cv2.line(img, origin, end, color, self._width)
        points = self._compute_arrow_points(origin, end)
        if points is None:
            return
        C, D = points
        cv2.line(img, end, C, color, self._width)
        cv2.line(img, end, D, color, self._width)

    def _compute_arrow_points(self, (xa, ya), (xb, yb), length=20.):
        # The arrow tip is made by joining B (xb, yb) to C and D. This method
        # computes the coordinates of C and D.
        ab_distance = math.sqrt( (xb - xa)**2 + (yb-ya)**2)
        if ab_distance == 0.:
            return None
        cos_beta = (xa - xb) / ab_distance
        sin_beta = (ya - yb) / ab_distance
        cos_alpha = self.cos_alpha
        sin_alpha = self.sin_alpha
        xc = xb + length * (cos_alpha * cos_beta - sin_alpha * sin_beta)
        yc = yb + length * (sin_beta * cos_alpha + sin_alpha * cos_beta)
        xd = xb + length * (cos_alpha * cos_beta + sin_alpha * sin_beta)
        yd = yb + length * (sin_beta * cos_alpha - sin_alpha * cos_beta)

        return ((int(xc), int(yc)), (int(xd), int(yd)))


class OpticalFlowDrawer(OpticalFlowMuxer):
    __gstdetails__ = ("Optical flow drawer",
                    "Filter/Video",
                    "draw optical flow on frames",
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

    line_thickness = gobject.property(type=int,
                                      default=2,
                                      blurb='thickness of the lines used to draw the arrow')

    def __init__(self, *args, **kw):
        super(OpticalFlowDrawer, self).__init__(*args, **kw)

        self.srcpad = gst.Pad(self.src_template)
        self.add_pad(self.srcpad)


        self._drawer = ArrowDrawer(width=self.line_thickness)


    def mux(self, buf, flow):
        if flow is None:
            return self.srcpad.push(buf)
        origins, ends = flow

        img = img_of_buf(buf)

        self._drawer.draw_arrows(img, origins, ends)

        new_buf = buf_of_img(img, bufmodel=buf)

        return self.srcpad.push(new_buf)


gobject.type_register (OpticalFlowDrawer)
ret = gst.element_register (OpticalFlowDrawer, 'opticalflowdrawer')
