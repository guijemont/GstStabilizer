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
        features = cv.GoodFeaturesToTrack(img, eigImage, tempImage, MAX_COUNT, 0.01, 10)

        return cv.FindCornerSubPix(img, features, (10, 10), (-1, -1),
                                   (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS,
                                    20, 0.03))

    def _filter_features(self, features, flter):
        return [ p for (status,p) in zip(flter, features)]

    def optical_flow(self, img0, img1):
        """
        Return two sets of coordinates (c0, c1), in img0 and img1 respectively,
        such that c1[i] is the position in img1 of the feature that is at c0[i]
        in img0.
        """
        corners0 = self._features(img0)

        corners1, status, _ = cv.CalcOpticalFlowPyrLK (
                     img0, img1, None, None,
                     corners0,
                     (10, 10), 3,
                     (cv.CV_TERMCRIT_ITER|cv.CV_TERMCRIT_EPS, 20, 0.03),
                     0)

        corners0 = self._filter_features(corners0, status)
        corners1 = self._filter_features(corners1, status)

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
    """
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


class OpticalFlowDrawer(SuccessiveImageTransformer):
    def __init__(self, *args, **kw):
        super(OpticalFlowDrawer, self).__init__(*args, **kw)
        self._flow_finder = OpticalFlowFinder()
        self._drawer = ArrowDrawer()

    def transform(self, img0, img0_transformed, img1):
        origins, ends = self._flow_finder.optical_flow(img0, img1)

        img_result = copy_img(img1)
        self._drawer.draw_arrows(img_result, origins, ends)

        return img_result


class HomographyFinder(object):
    def __init__(self, *args, **kw):
        super(HomographyFinder, self).__init__(*args, **kw)
        self._img0 = None
        self._flow_finder = OpticalFlowFinder()

    def _mat_of_point_list(self, points):
        n = len(points)
        mat = cv.CreateMat(1, n, cv.CV_32FC2)
        for i in xrange(n):
          mat[0, i] = points[i]
        return mat

    def _homography(self, points0, points1):
        mat0 = self._mat_of_point_list (points0)
        mat1 = self._mat_of_point_list (points1)
        homography = cv.CreateMat(3, 3, cv.CV_64F)

        # This is the homography to go form mat0 to mat1. We will pass it to
        # WarpPerspective() with img1, so that we can get it in the same configuration as img0
        cv.FindHomography (mat0, mat1, homography)

        return homography

    def _img_to_buf(self, img, bufmodel=None):
        buf = gst.Buffer(img.tostring())
        if bufmodel is not None:
            buf.caps = bufmodel.caps
            buf.duration = bufmodel.duration
            buf.timestamp = bufmodel.timestamp
            buf.offset = bufmodel.offset
            buf.offset_end = bufmodel.offset_end
        return buf

    def _new_image(self, width, height):
        """
        Create a new 8 bit 1 plane grayscale image of width and height,
        intialised with 0.
        """
        new_data = '\0' * width * height;
        new_img = cv.CreateImageHeader((width, height), 8, 1)
        cv.SetData(new_img, new_data)
        return new_img

    def new_img(self, img1):
        """
        Returns the homography with previous image, or None if it's the first
        image.
        """
        h = None
        if self._img0 is not None:
            corners0, corners1 = self._flow_finder.optical_flow(self._img0, img1)
            h = self._homography (corners0, corners1)


        self._img0 = img1

        return h

    def transform_buf(self, buf):
        struct = buf.caps[0]
        width = struct['width']
        height = struct['height']
        channels = 1    # we only accept x-raw-gray for now
        depth = 8       # depth=8 also in the caps
        img = cv.CreateImageHeader((width, height), depth, channels);
        cv.SetData (img, buf.data)

        new_img = self.transform_img(img)

        if new_img:
            return self._img_to_buf(new_img, bufmodel=buf)

        return None

    def transform_img(self, img):
        h = self.new_img(img)
        if h is not None:
            # this should be the general case (not first image): we have an
            # already "stabilised" image in ._img0. We "stabilise" img
            # relatively to it (h is the homography to go from ._img0 to img).
            self._img0 = self._apply_homography(h, img)
            return self._img0

        return img

    def _apply_homography(self, homography, img):
        new_img = self._new_image(*(cv.GetSize(img)))
        cv.WarpPerspective(img, new_img, homography, cv.CV_WARP_INVERSE_MAP)

        return new_img


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

    self._mem_storage = cv.CreateMemStorage()

    self._first_keypoints = None
    self._first_descriptors = None

    self._last_buf = None
    # "backwards" points for _last_buf, and whether they should be displayed as
    # crosses
    self._last_buf_data = ([], False)

    self._h_finder = HomographyFinder()

    self._flow_drawer = OpticalFlowDrawer()

  def _buf_to_cv_img(self, buf):
    struct = buf.caps[0]
    width = struct['width']
    height = struct['height']
    channels = 1    # we only accept x-raw-gray for now
    depth = 8       # depth=8 also in the caps
    img = cv.CreateImageHeader((width, height), depth, channels);
    cv.SetData (img, buf.data)

    return img

  def _euclidian_distance(self, vector1, vector2, maximum=None):
    # if this is too slow, we could use numpy to do the job:
    # http://stackoverflow.com/questions/1401712/calculate-euclidean-distance-with-numpy
    d = 0
    for x1,x2 in izip(vector1, vector2):
      sqr_d = x1 - x2
      d += sqr_d * sqr_d
      if maximum is not None and d > maximum:
        # we're not interested in distances that are bigger than
        # maximum, stop computing it.
        break

    return d

  def _naive_nearest_neighbor (self, keypoint, descriptor):
    shortest = None
    second_shortest = None
    neighbor = None
    for original_kp, original_desc in  \
        izip(self._first_keypoints, self._first_descriptors):
      # compare laplacians, stored in tuple element 1
      if keypoint[1] != original_kp[1]:
        continue
      d = self._euclidian_distance(descriptor, original_desc,
                                   second_shortest)
      if shortest is None or d < shortest:
        second_shortest = shortest
        shortest = d
        neighbor = original_kp
      elif second_shortest is None or d < second_shortest:
        second_shortest = d

    if shortest < self.SECOND_SHORTEST_RATIO * second_shortest:
      return neighbor

    return None

  def _find_pairs_from_keypoints (self, keypoints, descriptors):
    original_plane, frame_plane = [], []
    for kp,desc in izip(keypoints, descriptors):
      neighbor_kp = self._naive_nearest_neighbor (kp, desc)
      if neighbor_kp is not None:
        original_plane.append(neighbor_kp[0])
        frame_plane.append(kp[0])

    return original_plane, frame_plane

  def _print_matrix(self, mat):
    for i in xrange(mat.rows):
      for j in xrange(mat.cols):
        print "% 13.8f" % mat[i, j],
      print

  def _find_planes (self, img):
    keypoints, descriptors = cv.ExtractSURF(img, None,
                                            self._mem_storage,
                                            (0, 9000, 3, 1))
    print "Got %d key points and %d descriptors" % (len(keypoints),
                                                    len(descriptors))
    if self._first_keypoints is None:
      self._first_keypoints = keypoints
      self._first_descriptors = descriptors
      return None
    else:
      original_plane, frame_plane = self._find_pairs_from_keypoints (keypoints, descriptors)
      return original_plane, frame_plane

  def _find_homography(self, img):
    planes = self._find_planes (img)
    if planes is None:
      return None

    original_plane, frame_plane = planes
    n = len(original_plane)
    print "found %d pairs" % n
    orig_mat = cv.CreateMat(1, n, cv.CV_32FC2)
    for i in xrange(n):
      orig_mat[0, i] = original_plane[i]
    frame_mat = cv.CreateMat(1, n, cv.CV_32FC2)
    for i in xrange(n):
      frame_mat[0, i] = frame_plane[i]
    homography = cv.CreateMat(3, 3, cv.CV_64F)
    # we get the homography to go from the original frame to the
    # current frame. That's the inverse of the homography we want to
    # apply, which is what cv.WarpPerspective() likes most
    cv.FindHomography (orig_mat, frame_mat, homography)
    print "found homography:"
    self._print_matrix (homography)
    return homography

  def _new_image(self, width, height):
    """
    Create a new 8 bit 1 plane grayscale image of width and height,
    intialised with 0.
    """
    new_data = '\0' * width * height;
    new_img = cv.CreateImageHeader((width, height), 8, 1)
    cv.SetData(new_img, new_data)
    return new_img

  def _img_to_buf(self, img, bufmodel=None):
    buf = gst.Buffer(img.tostring())
    if bufmodel is not None:
        buf.caps = bufmodel.caps
        buf.duration = bufmodel.duration
        buf.timestamp = bufmodel.timestamp
        buf.offset = bufmodel.offset
        buf.offset_end = bufmodel.offset_end
    return buf

  def _draw_shape(self, img, point, shape):
    x,y = point
    size = 10
    def draw_line(P1, P2):
      cv.Line(img, P1, P2, (128,), 2)
    if shape == SHAPE_CROSS:
      draw_line((x-size, y-size), (x+size, y+size))
      draw_line((x-size, y+size), (x+size, y-size))
    elif shape == SHAPE_BOX:
      draw_line((x-size, y-size), (x+size, y-size))
      draw_line((x+size, y-size), (x+size, y+size))
      draw_line((x+size, y+size), (x-size, y+size))
      draw_line((x-size, y+size), (x-size, y-size))
    elif shape == SHAPE_CIRCLE:
      cv.Circle(img, point, size, (128,), 2)

  def _show_points(self, img, points, shape):
    for point in points:
      self._draw_shape(img, point, shape)

  def _show_points_buf(self, buf, points, shape):
    img = self._buf_to_cv_img(buf)
    self._show_points(img, points, shape)
    return self._img_to_buf(img, buf)

  def _apply_homography(self, homography, buf, img):
    new_img = self._new_image((buf.caps[0]['width'], buf.caps[0]['height']))
    cv.WarpPerspective(img, new_img, homography, cv.CV_WARP_INVERSE_MAP)
    newbuf = gst.Buffer(new_data)
    newbuf.caps = buf.caps
    newbuf.duration = buf.duration
    newbuf.timestamp = buf.timestamp
    newbuf.offset = buf.offset
    newbuf.offset_end = buf.offset_end
    return newbuf

  def chain(self, pad, buf):
    img = img_of_buf(buf)
    transformed_img = self._flow_drawer.process_image(img)
    new_buf = buf_of_img(transformed_img, bufmodel=buf)

    return self.srcpad.push(new_buf)


    img = self._buf_to_cv_img (buf)
    planes = self._find_planes (img)
    #homography = self._find_homography(img)
    #if homography:
    if planes:
      #newbuf = self._apply_homography(homography, buf, img)
      previous_plane, current_plane = planes
      print "Got %d matches" % len(previous_plane)
    previous_frame = self._last_buf
    self._last_buf = buf
    if previous_frame:
      # note: we're about to display the previous frame and just queued the
      # current one
      previous_frame_next_points, current_frame_previous_points = planes
      previous_frame_previous_points, previous_shape = self._last_buf_data
      current_shape = (previous_shape + 1) % SHAPE_MODULO
      self._last_buf_data = current_frame_previous_points, current_shape
      if previous_frame_previous_points:
        previous_frame = self._show_points_buf(previous_frame, previous_frame_previous_points, previous_shape)
      previous_frame = self._show_points_buf(previous_frame, previous_frame_next_points, current_shape)
      return self.srcpad.push(previous_frame)
    else:
      return gst.FLOW_OK



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
