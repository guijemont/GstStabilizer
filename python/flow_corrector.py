import gobject,gst

import cv

from cv_gst_util import *

from flow_muxer import OpticalFlowMuxer

import cv_flow_finder


class OpticalFlowCorrector(gst.Element):
    __gstdetails__ = ("Optical flow corrector",
                    "Filter/Video",
                    "Correct frames according to global optical flow so as to invert it ('stabilise' images)",
                    "Guillaume Emont")
    sink_template = gst.PadTemplate ("sink",
                                      gst.PAD_SINK,
                                      gst.PAD_ALWAYS,
                                      gst.Caps('video/x-raw-rgb,depth=24'))
    src_template = gst.PadTemplate("src",
                                    gst.PAD_SRC,
                                    gst.PAD_ALWAYS,
                                    gst.Caps('video/x-raw-rgb,depth=24'))
    __gsttemplates__ = (sink_template, src_template)

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

    def __init__(self, *args, **kw):
        super(OpticalFlowCorrector, self).__init__(*args, **kw)

        self.srcpad = gst.Pad(self.src_template)
        self.add_pad(self.srcpad)

        self.sinkpad = gst.Pad(self.sink_template)
        self.sinkpad.set_chain_function(self._chain)
        self.add_pad(self.sinkpad)

        self._reference_img = None

        self._finder = cv_flow_finder.Finder(self.corner_count,
                                             self.corner_quality_level,
                                             self.corner_min_distance,
                                             self.win_size,
                                             self.pyramid_level,
                                             self.max_iterations,
                                             self.epsilon)

    def _chain(self, pad, buf):
        if self._reference_img is None:
            self._reference_img = img_of_buf(buf)
            return self.srcpad.push(buf)

        flow = self._get_flow(buf)
        if flow is None:
            return self.srcpad.push(buf)

        try:
            transform = self._perspective_transform_from_flow(flow)

            img = img_of_buf(buf)
            new_img = cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_8U, 3)
            cv.WarpPerspective(img, new_img, transform, cv.CV_WARP_INVERSE_MAP)

            new_buf = buf_of_img(new_img, bufmodel=buf)
            self._reference_img = img_of_buf(new_buf)
            return self.srcpad.push(new_buf)
        except cv.error,e :
            print "got an opencv error (%s), not applying any transform for this frame" % e.message
            self._reference_img = img_of_buf(buf)
            return self.srcpad.push(buf)

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

    def _get_flow(self, buf):

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

        color_img = img_of_buf(buf)
        gray_img = gray_scale(color_img)
        gray_ref_img = gray_scale(self._reference_img)

        return self._finder.optical_flow_img(gray_ref_img, gray_img)

    def _has_ignore_box(self):
        return (-1) not in (self.ignore_box_min_x, self.ignore_box_max_x,
                            self.ignore_box_min_y, self.ignore_box_max_y)


gobject.type_register (OpticalFlowCorrector)
ret = gst.element_register (OpticalFlowCorrector, 'opticalflowcorrector')
