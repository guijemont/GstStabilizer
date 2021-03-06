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
        self.flow_sink_pad.set_event_function(self._flow_event)
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

    def _flow_event(self, pad, event):
        # We just drop all new segment events from the flow pad. We assume they
        # are only duplicates of those we got on main_sink_pad (which are
        # automatically forwarded since we have no event handling function)
        # This might be dirty but seems to be working.
        if event.type == gst.EVENT_NEWSEGMENT:
            return True
        return False


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
