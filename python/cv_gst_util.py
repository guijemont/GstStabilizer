#!/usr/bin/env python

import cv2, gst, numpy

# note that we only care about what OpticalFlowCorrector supports
def img_of_buf(buf):
    if buf is None:
        return None
    struct = buf.caps[0]
    width = struct['width']
    height = struct['height']
    if struct.has_field('bpp'):
        # yeah, we only support 8 bits per channel
        channels = struct['bpp'] / 8
        depth = 8
    array = numpy.frombuffer(buf, dtype=numpy.uint8)
    img = array.reshape((height, width, channels))

    return img

def buf_of_img(img, bufmodel=None):
    buf = gst.Buffer(img)
    if bufmodel is not None:
        buf.caps = bufmodel.caps
        buf.duration = bufmodel.duration
        buf.timestamp = bufmodel.timestamp
        buf.offset = bufmodel.offset
        buf.offset_end = bufmodel.offset_end
    return buf

def gray_scale(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return new_img

def green_component(img):
    new_img = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
    cv.Split(img, None, new_img, None, None)
    return new_img

def numpy_to_iplimg(image):
    ipl_image = cv.CreateImageHeader((image.shape[1], image.shape[0]),
                                     image.dtype.itemsize * 8,
                                     image.shape[2])
    cvmat = cv.fromarray(image)
    cv.SetData(ipl_image, cvmat.tostring())
    return ipl_image
