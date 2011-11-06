#!/usr/bin/env python
import gst

import cPickle
from collections import deque


class OpticalFlowMuxer(gst.Element):
    """
    Base class for "muxers" of an optical flow stream and another stream having
    the same timestamps.
    When subclassing, you should implement mux and redefine main_sink_template
    """

    flow_sink_template = gst.PadTemplate ("flowsink",
                                           gst.PAD_SINK,
                                           gst.PAD_ALWAYS,
                                           gst.Caps('application/x-motion-flow'))

    # should be defined as a proper pad template in the subclass
    main_sink_template = None

    def __init__(self):
        gst.Element.__init__(self)

        self.flow_sink_pad = gst.Pad(self.flow_sink_template)
        self.flow_sink_pad.set_chain_function(self._chain)
        self.add_pad(self.flow_sink_pad)

        self.main_sink_pad = gst.Pad(self.main_sink_template)
        self.main_sink_pad.set_chain_function(self._chain)
        self.add_pad(self.main_sink_pad)

        # FIXME: shouldn't we just use gstreamer queues outside of the elemnt?
        self._pending_flow = deque()
        self._pending_main = deque()

    def mux(self, buf, flow):
        raise NotImplementedError("This method needs to be implemented in a subclass")

    def _chain(self, pad, buf):
        if pad == self.flow_sink_pad:
            self._pending_flow.append(buf)
        else: # self.main_sink_pad
            self._pending_main.append(buf)
        return self._try_mux()

    def _try_mux(self):
        if self._pending_flow and self._pending_main:
            flow_buf = self._pending_flow.popleft()
            flow = cPickle.loads(flow_buf.data)
            buf = self._pending_main.popleft()
            if buf.timestamp == flow_buf.timestamp:
                return self.mux(buf, flow)
            else:
                print "All going wrong!"
                return gst.FLOW_ERROR

        return gst.FLOW_OK


class TestFlowMuxer(OpticalFlowMuxer):
    __gstdetails__ = ("Optical flow muxer",
                    "Filter/Video",
                    "test for the base OpticalFLowMuxer class",
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
        super(TestFlowMuxer, self).__init__(*args, **kw)

        self.srcpad = gst.Pad(self.src_template)
        self.add_pad(self.srcpad)

    def mux(self, buf, flow):
        ret = self.srcpad.push(buf)
        return ret


import gobject
gobject.type_register (TestFlowMuxer)
ret = gst.element_register (TestFlowMuxer, 'testflowmuxer')
