#!/usr/bin/env python3

# USAGE : python3 <script_name> <jpg filepath>
# ex. python3 number_plate_image_final.py ./02.JPG

# Pipeline input_image -> LPR -> Output

################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

from os import path
import os.path
import os 
import cv2
import gi
import numpy as np
import platform
import math
import time
from ctypes import *
gi.require_version('Gst', '1.0')
from gi.repository import GLib
from gi.repository import GObject, Gst
import sys
sys.path.append('../')
from common.FPS import GETFPS
from common.bus_call import bus_call
from common.is_aarch_64 import is_aarch64
import pyds


fps_streams = {}
upper_text = ""
lower_text = ""
MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 0
pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]


detections = {}

# tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.


def tiler_sink_pad_buffer_probe(pad, info, u_data):
    global detections,upper_text,lower_text
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        # print("frame_no",frame_number)
        # print("Source id is ", frame_meta.source_id)
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        # print("Num object meta ", frame_meta.num_obj_meta)
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            
            c_list = obj_meta.classifier_meta_list
            while c_list is not None:
                try:
                    classifier_list_meta=pyds.NvDsClassifierMeta.cast(c_list.data) 
                except StopIteration:
                    break
                # print(classifier_list_meta.num_labels)

                label_list = classifier_list_meta.label_info_list
                # print("*************")
                while label_list is not None:
                    try:
                        label_list_meta=pyds.NvDsLabelInfo.cast(label_list.data) 
                    except StopIteration:
                        break
                    # print(label_list_meta.label_id)         
                    # print('------',label_list_meta.result_label)         
                    detections.update({frame_meta.source_id: {}})
                    detections[frame_meta.source_id] = label_list_meta.result_label
                    try: 
                        label_list=label_list.next
                    except StopIteration:
                        break
                # print("*************") 
                try: 
                    c_list=c_list.next
                except StopIteration:
                    break    
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

            # detections.append(copy.deepcopy(dictionary))
            # print(detections)
        else:
            if not frame_meta.source_id in detections:
                detections.update({frame_meta.source_id: {}})
                detections[frame_meta.source_id] = 'None'
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

        # writing in result.txt after all the detection
        # print("detection:",detections)
        if len(detections)>1:
            # print("+++++++++++++++",detections)
            
            f = open(os.getcwd() + "/result.txt", "w+")
            f.write(detections[0]+"\n")
            f.write(detections[1]+"\n")
            f.close()
            break


    return Gst.PadProbeReturn.OK

def draw_bounding_boxes(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[obj_meta.class_id]
    image = cv2.rectangle(image, (left, top),
                          (left+width, top+height), (0, 0, 255, 0), 2)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image = cv2.putText(image, obj_name+',C='+str(confidence),
                        (left-10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 0), 2)
    return image



def create_source_bin(index, uri):
    # print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%0d" % index
    # print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # source = Gst.ElementFactory.make("multifilesrc", "source")
    source = Gst.ElementFactory.make("filesrc", "source")
    jpegparser = Gst.ElementFactory.make("jpegparse", "jpeg-parser")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not source:
        sys.stderr.write(" Unable to create source \n")
    if not jpegparser:
        sys.stderr.write(" Unable to create jpegparser \n")
    if not decoder:
        sys.stderr.write(" Unable to create decoder \n")
    
    source.set_property("location", uri)
    # decoder.set_property("mjpeg", 1)
    
    # Gst.Bin.add(nbin, source,jpegparser,decoder)
    # Gst.Bin.link.many(source, jpegparser, decoder)
    nbin = Gst.Bin.new("image_sink_bin_%0d" %index)
    nbin.add(source)
    nbin.add(jpegparser)
    nbin.add(decoder)

    source.link(jpegparser)
    jpegparser.link(decoder)

    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get src pad \n")

    bin_ghost_pad = nbin.get_static_pad("src")
    if not bin_ghost_pad.set_target(srcpad):
        sys.stderr.write(
            "Failed to link decoder src pad to source bin ghost pad\n")
    return nbin


def main(args):
    # print(len(args))
    # if len(args) < 2:
    #     sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
    #     sys.exit(1)

    for i in range(0, len(args)-1):
        fps_streams["stream{0}".format(i)] = GETFPS(i)
        # print(GETFPS(i))
    number_sources = len(args)-1
    # print(number_sources)
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    # print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    # print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        # print("Creating source_bin ", i, " \n ")
        uri_name = args[i+1]
        # print("-----",uri_name)
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    # print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")


    # print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    # print("Creating nvvidconv \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create convertor1 \n")

    # print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    # nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    # nvosd.set_property('display-text',OSD_DISPLAY_TEXT)
    if(is_aarch64()):
        # print("Creating transform \n ")
        transform = Gst.ElementFactory.make(
            "nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")

    # print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("fakesink", "nvvideo-renderer")
    # sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        # print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "lpr_config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        # print("WARNING: Overriding infer-config batch-size",
        #       pgie_batch_size, " with number of sources ", number_sources, " \n")
        pgie.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    sink.set_property("sync", 0)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    # print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    # print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)

    # pgie.link(tiler)
    # tiler.link(nvvidconv)
    # nvvidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER,
                                tiler_sink_pad_buffer_probe, 0)

    # List the sources
    # print("Now playing...")    
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    # print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)
    


if __name__ == '__main__':
    sys.exit(main(sys.argv))
