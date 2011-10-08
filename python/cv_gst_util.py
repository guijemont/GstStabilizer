#!/usr/bin/env python

import cv, gst

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

def gray_scale(img):
    new_img = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
    cv.CvtColor(img, new_img, cv.CV_RGB2GRAY)
    return new_img

def resize(img, new_width, new_height):
    new_img = cv.CreateImage((new_width, new_height), img.depth, img.channels)
    cv.Resize(img, new_img)
    return new_img

