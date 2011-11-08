GstStabilizer
=============

This stuff is still in a rather early stage of development (or should I say
"research"), but might already be useful in some use cases.

You should be able to get the latest version like this::

  git clone git://gitorious.org/gststabilizer/gststabilizer.git


Dependencies
------------

 - Python (only tested with 2.7) http://python.org/
 - GStreamer 0.10 (only tested with rather recent versions, what a recent 
   distro has should do) http://gstreamer.freedesktop.org/
 - gst-python (same as above) http://gstreamer.freedesktop.org/
 - OpenCV >= 2.1.0, with the "new style" ``cv2`` python bindings compiled
   http://opencv.willowgarage.com/


Usage
-----

For the elements to be recognized, you need to point ``GST_PLUGIN_PATH``

Example pipeline::

  gst-launch filesrc location=<my_shaky_video> ! decodebin ! tee name=tee \
    tee. ! ffmpegcolorspace ! opticalflowfinder ! opticalflowrevert name=mux \
    tee. ! ffmpegcolorspace ! mux. \
    mux. ! ffmpegcolorspace ! xvimagesink

Note that depending on the video and the options you give to
``opticalflowfinder``, live stabilisation might not always be doable. If it's
too laggy, your probably want to encode and save the stream instead of sending
it to a visualisation sink.

You want to have a look at the myriad of options that can be set in ``opticalflowfinder``::

  gst-inspect opticalflowfinder

The most important of them is the algorithm, the two currently implemented are:

Lucas-Kanade
  The faster one, good for typical video streams where there is little change
  from one frame to the next.
SURF
  Slower, but can handle big changes from one frame to the next. Very useful
  for time lapses taken from a moving camera (specially developed for a time
  lapse from a tethered helium balloon, see http://balloonfreaks.mooo.com/).

Limitations
-----------
 - Only works if the original stream always points towards the same area of
   interest, i.e. does not support voluntary camera movements, they are all
   recognized as "shake" that should be compensated.
 - Slower than what it could actually be


Bugs and patches
----------------

Over there: https://github.com/guijemont/GstStabilizer/issues
