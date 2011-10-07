import gobject,gst

import cv

from cv_gst_util import *

from flow_muxer import OpticalFlowMuxer


class OpticalFlowCorrector(OpticalFlowMuxer):
    __gstdetails__ = ("Optical flow corrector",
                    "Filter/Video",
                    "Correct frames according to global optical flow so as to invert it ('stabilise' images)",
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


    def __init__(self, *args, **kw):
        super(OpticalFlowCorrector, self).__init__(*args, **kw)

        self.srcpad = gst.Pad(self.src_template)
        self.add_pad(self.srcpad)

    def mux(self, buf, flow):
        if flow is None:
            return self.srcpad.push(buf)

        transform = self._perspective_transform_from_flow(flow)

        img = img_of_buf(buf)
        new_img = cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_8U, 3)
        cv.WarpPerspective(img, new_img, transform, cv.CV_WARP_INVERSE_MAP)

        new_buf = buf_of_img(new_img, bufmodel=buf)
        return self.srcpad.push(new_buf)

    def _mat_of_point_list(self, points):
        n = len(points)
        mat = cv.CreateMat(1, n, cv.CV_32FC2)
        for i in xrange(n):
          mat[0, i] = points[i]
        return mat

    def _perspective_transform_from_flow(self, (points0, points1)):
        mat0 = self._mat_of_point_list (points0)
        mat1 = self._mat_of_point_list (points1)
        transform = cv.CreateMat(3, 3, cv.CV_64F)

        cv.FindHomography(mat0, mat1, transform)

        return transform

gobject.type_register (OpticalFlowCorrector)
ret = gst.element_register (OpticalFlowCorrector, 'opticalflowcorrector')
