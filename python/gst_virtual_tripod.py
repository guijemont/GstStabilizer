#!/usr/bin/env python
import glib,gobject, gst

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

    def __init__(self):
        gst.Element.__init__(self)

        self.sinkpad = gst.Pad(self.sink_template)
        self.sinkpad.set_chain_function(self.chain)
        self.add_pad(self.sinkpad)

        self.srcpad = gst.Pad(self.src_template)
        self.add_pad(self.srcpad)


    def _buf_to_cv_img(self, buf):
        struct = buf.caps[0]
        width = struct['width']
        height = struct['height']
        channels = 1    # we only accept x-raw-gray for now
        depth = 8       # depth=8 also in the caps
        img = cv.CreateImageHeader((width, height), depth, channels);
        cv.SetData (img, buf.data)

        return img


    def chain(self, pad, buf):
        print "Got buffer:", repr(buf)
        return self.srcpad.push(buf)


gobject.type_register (VirtualTripod)
ret = gst.element_register (VirtualTripod, 'virtualtripod')

def main(args):
    pipeline = gst.parse_launch("videotestsrc ! virtualtripod ! fakesink")
    pipeline.set_state (gst.STATE_PLAYING)
    gobject.MainLoop().run()

if __name__ == '__main__':
    import sys
    glib.threads_init()
    main(sys.argv)

__gstelementfactory__ = ( "virtualtripod", gst.RANK_MARGINAL, VirtualTripod)
