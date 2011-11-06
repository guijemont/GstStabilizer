import gst, gobject

from flow_muxer import OpticalFlowMuxer
from cv_gst_util import *


class OpticalFlowRevert(OpticalFlowMuxer):
    __gstdetails__ = ("Optical flow revert",
                    "Filter/Video",
                    "Revert the effect of the flow",
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
        super(OpticalFlowRevert, self).__init__(*args, **kw)

        self.srcpad = gst.Pad(self.src_template)
        self.add_pad(self.srcpad)

        self._last_output_img = None
        self._reference_transform = numpy.asarray([[1., 0., 0.],
                                                   [0., 1., 0.],
                                                   [0., 0., 1.]],
                                                   dtype=numpy.float128)

    def mux(self, buf, flow):
        if flow is None:
            self._last_output_img = img_of_buf(buf)
            return self.srcpad.push(buf)

        origins, ends = flow

        transform, mask = cv2.findHomography(origins, ends,
                                             method=cv2.RANSAC,
                                             ransacReprojThreshold=3)

        # we accumulate the transformations, so that we apply a transformation
        # relative to the first frame
        self._reference_transform = transform.dot(self._reference_transform)

        img = img_of_buf(buf)
        new_img = self._last_output_img.copy()

        new_img = cv2.warpPerspective(img,
                                      numpy.asarray(self._reference_transform,
                                                    dtype=numpy.float64),
                                      (img.shape[1], img.shape[0]),
                                      dst=new_img,
                                      flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT)

        self._last_output_img = new_img
        new_buf = buf_of_img(new_img, bufmodel=buf)
        return self.srcpad.push(new_buf)


gobject.type_register (OpticalFlowRevert)
ret = gst.element_register (OpticalFlowRevert, 'opticalflowrevert')
