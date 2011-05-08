#!/usr/bin/env python
import glib,gobject,gst,sys

from itertools import izip

import cv

SHAPE_CROSS = 0
SHAPE_BOX = 1
SHAPE_CIRCLE = 2

SHAPE_MODULO = 3

MAX_COUNT = 500

#
# Algo idea:
#
# - get features of img0
# - get optical flow (2 set of corresponding coordinates)
# - compute homography from these
#

def img_of_buf(buf):
    struct = buf.caps[0]
    width = struct['width']
    height = struct['height']
    channels = 1    # we only accept x-raw-gray for now
    depth = 8       # depth=8 also in the caps
    img = cv.CreateImageHeader((width, height), depth, channels);
    cv.SetData (img, buf.data)
    return img

def buf_of_img(img, bufmodel=None):
    buf = gst.Buffer(img.tostring())
    if bufmodel is not None:
        buf.caps = bufmodel.caps
        buf.duration = bufmodel.duration
        buf.timestamp = bufmodel.timestamp
        buf.offset = bufmodel.offset
        buf.offset_end = bufmodel.offset_end
    return buf

def copy_img(img):
    new_img = cv.CreateImageHeader((img.width, img.height), 8, 1)
    cv.SetData(new_img, img.tostring())
    return new_img

class OpticalFlowFinder(object):
    def _features(self, img):
        img_size = cv.GetSize(img)
        eigImage = cv.CreateImage(img_size, cv.IPL_DEPTH_8U, 1)
        tempImage = cv.CreateImage(img_size, cv.IPL_DEPTH_8U, 1)
        features = cv.GoodFeaturesToTrack(img, eigImage, tempImage,
                                          MAX_COUNT, #number of corners to detect
                                          0.1, #Multiplier for the max/min
                                                #eigenvalue; specifies the minimal
                                                #accepted quality of image corners
                                          30 # minimum distance between returned corners
                                          )

        return cv.FindCornerSubPix(img, features, (10, 10), (-1, -1),
                                   (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS,
                                    20, 0.03))

    def _filter_features(self, features, flter, errors):
        filtered_errors = [err for (err, status) in zip(errors, flter) if status]
        n = len(filtered_errors) / 2 # we want to keep the best third only
        admissible_error = sorted(filtered_errors)[n]
        return [ p for (status,p, err) in zip(flter, features, errors)
                    if status and err < admissible_error]

    def optical_flow(self, img0, img1):
        """
        Return two sets of coordinates (c0, c1), in img0 and img1 respectively,
        such that c1[i] is the position in img1 of the feature that is at c0[i]
        in img0.
        """
        corners0 = self._features(img0)

        corners1, status, track_errors = cv.CalcOpticalFlowPyrLK (
                     img0, img1, None, None,
                     corners0,
                     (30, 30), # win size
                     3, # pyramid level
                     (cv.CV_TERMCRIT_ITER|cv.CV_TERMCRIT_EPS, # stop type
                      20, # max iterations
                      0.01), # min accuracy
                     0) # flags

        corners0 = self._filter_features(corners0, status, track_errors)
        corners1 = self._filter_features(corners1, status, track_errors)

        return (corners0, corners1)

class ArrowDrawer(object):
    def draw_arrows(self, img, origins, ends):
        for origin, end in zip(origins, ends):
            self.draw_arrow(img, origin, end)

    def draw_arrow(self, img, origin, end):
        cv.Line(img, origin, end, (128,), 2)
        # yeah, that's a lazy approximation of an arrow
        cv.Circle(img, end, 4, (128,), 2)

class SuccessiveImageTransformer(object):
    """
    Apply transformations on images that depend on the previous image (either
    raw or transformed).
    You want to subclass transform().

    If reference_is_first is True, the transformation will be made compared to
    the first image of the series, not the previous one.
    """
    reference_is_first = False
    def __init__(self, *args, **kw):
        super(SuccessiveImageTransformer, self).__init__(*args, **kw)
        self._img0 = None
        self._img0_transformed = None

    def process_image(self, img):
        """
        Will return img, as (optionally) transformed by transform(), and will
        keep a reference on both img and its transformed value until the next
        call to process_image().
        """
        if self._img0:
            img_transformed = self.transform(self._img0,
                                             self._img0_transformed,
                                             img)
        else:
            img_transformed = None

        if self._img0 is None or self.reference_is_first == False:
            self._img0, self._img0_transformed = img, img_transformed

        if img_transformed:
            return img_transformed
        else:
            return img

    def transform(self, img0, img0_transformed, img1):
        """
        At this point, img0 and img0_transformed are the previous
        image before and after transformation (the latter might be None),
        img1 is the new image, and you should return a transformed version
        """
        raise NotImplementedError("transform() needs to be implemented in a subclass")

class ReferenceImageTransformer(SuccessiveImageTransformer):
    reference_is_first = True

class OpticalFlowDrawer(ReferenceImageTransformer):
    def __init__(self, *args, **kw):
        super(OpticalFlowDrawer, self).__init__(*args, **kw)
        self._flow_finder = OpticalFlowFinder()
        self._drawer = ArrowDrawer()

    def transform(self, img0, img0_transformed, img1):
        origins, ends = self._flow_finder.optical_flow(img0, img1)

        img_result = copy_img(img1)
        self._drawer.draw_arrows(img_result, origins, ends)

        return img_result


class VirtualTripod(gst.Element):
  __gstdetails__ = ("virtual tripod",
                    "Filter/Video",
                    "Align your frames",
                    "Guillaume Emont")

  sink_template = gst.PadTemplate ("sink",
                                   gst.PAD_SINK,
                                   gst.PAD_ALWAYS,
                                   gst.Caps('video/x-raw-gray,depth=8'))

  src_template = gst.PadTemplate ("source",
                                     gst.PAD_SRC,
                                     gst.PAD_ALWAYS,
                                     gst.Caps('video/x-raw-gray,depth=8'))

  __gsttemplates__ = (sink_template, src_template)

  # value taken from opencv's find_obj.cpp sample
  SECOND_SHORTEST_RATIO = 0.4

  def __init__(self):
    gst.Element.__init__(self)

    self.sinkpad = gst.Pad(self.sink_template)
    self.sinkpad.set_chain_function(self.chain)
    self.add_pad(self.sinkpad)

    self.srcpad = gst.Pad(self.src_template)
    self.add_pad(self.srcpad)

    self._flow_drawer = OpticalFlowDrawer()


  def chain(self, pad, buf):
    img = img_of_buf(buf)
    transformed_img = self._flow_drawer.process_image(img)
    new_buf = buf_of_img(transformed_img, bufmodel=buf)

    return self.srcpad.push(new_buf)


gobject.type_register (VirtualTripod)
ret = gst.element_register (VirtualTripod, 'virtualtripod')

def main(args):
  pipeline = gst.parse_launch("filesrc location=/home/guijemont/Photos/tests/test_guij.ogg ! decodebin ! ffvideoscale  ! ffmpegcolorspace ! video/x-raw-gray ! virtualtripod ! ffmpegcolorspace ! xvimagesink")
  pipeline.set_state (gst.STATE_PLAYING)
  gobject.MainLoop().run()

if __name__ == '__main__':
  import sys
  glib.threads_init()
  main(sys.argv)

__gstelementfactory__ = ( "virtualtripod", gst.RANK_MARGINAL, VirtualTripod)
