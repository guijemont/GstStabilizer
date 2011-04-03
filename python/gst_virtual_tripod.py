#!/usr/bin/env python
import glib,gobject,gst,sys

from itertools import izip

import cv


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
                                                (0, 5000, 3, 1))
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
        img = self._buf_to_cv_img (buf)
        planes = self._find_planes (img)
        #homography = self._find_homography(img)
        #if homography:
        if planes:
          #newbuf = self._apply_homography(homography, buf, img)
          previous_plane, current_plane = planes
          print "Got %d matches" % len(previous_plane)
        buf_to_push = self._last_buf
        self._last_buf = buf
        if buf_to_push:
          return self.srcpad.push(buf_to_push)
        else:
          return gst.FLOW_OK



gobject.type_register (VirtualTripod)
ret = gst.element_register (VirtualTripod, 'virtualtripod')

def main(args):
    pipeline = gst.parse_launch("filesrc location=/home/guijemont/Photos/tests/test_guij.ogg ! decodebin ! ffvideoscale  ! ffmpegcolorspace ! video/x-raw-gray,width=320,height=240 ! virtualtripod ! ffmpegcolorspace ! xvimagesink")
    pipeline.set_state (gst.STATE_PLAYING)
    gobject.MainLoop().run()

if __name__ == '__main__':
    import sys
    glib.threads_init()
    main(sys.argv)

__gstelementfactory__ = ( "virtualtripod", gst.RANK_MARGINAL, VirtualTripod)
