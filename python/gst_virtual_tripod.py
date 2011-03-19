#!/usr/bin/env python
import glib,gobject, gst


class VirtualTripod(gst.Element):
    __gstdetails__ = ("virtual tripod",
                      "Filter/Video",
                      "Align your frames",
                      "Guillaume Emont")

    sink_template = gst.PadTemplate ("sink",
                                     gst.PAD_SINK,
                                     gst.PAD_ALWAYS,
                                     gst.Caps('video/x-raw-rgb'))

    src_template = gst.PadTemplate ("source",
                                       gst.PAD_SRC,
                                       gst.PAD_ALWAYS,
                                       gst.Caps('video/x-raw-rgb'))

    __gsttemplates__ = (sink_template, src_template)

    def __init__(self):
        gst.Element.__init__(self)

        self.sinkpad = gst.Pad(self.sink_template)
        self.sinkpad.set_chain_function(self.chain)
        self.add_pad(self.sinkpad)

        self.srcpad = gst.Pad(self.src_template)
        self.add_pad(self.srcpad)



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
