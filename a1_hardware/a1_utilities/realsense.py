import pyrealsense2 as rs
import os
import numpy as np
import cv2
import datetime
import time
import threading
from a1_utilities.logger import StateLogger
from a1_utilities.logger import VisualLogger


RGB_WIDTH = 320
RGB_HEIGHT = 180
RGB_RATE = 30

DEPTH_WIDTH = 424
DEPTH_HEIGHT = 240
DEPTH_RATE = 30


def process_depth(depth, target_shape):
  depth_dim = depth.shape
  horizontal_clip = int(depth_dim[0]/10)
  resized_depth_image = cv2.resize(
    depth[:,horizontal_clip:],
    dsize=target_shape,
    interpolation=cv2.INTER_NEAREST
  )
  blurred_depth_image = cv2.medianBlur(resized_depth_image, 3)
  return blurred_depth_image


# We do not use RGB rightnow
def process_rgb(rgb):
  return rgb

class A1RealSense:
  '''
  A1RealSense:
    wrapper of the pyrealsense functions for streaming RealSense Reading
  '''
  def __init__(
    self,
    target_shape=(64, 64),
    use_depth=True,
    use_rgb=False,
    save_frames=False,
    save_dir_name=None,
  ):
    self.pipeline = rs.pipeline()

    self.use_depth = use_depth
    self.use_rgb = use_rgb

    self.target_shape = target_shape
    if self.use_rgb:
      self.rgb_frame = np.zeros(target_shape + (3,)) #Where we keep track of rgbd frame
    if self.use_depth:
      self.depth_frame = np.zeros(target_shape + (1,)) #Where we keep track of rgbd frame

    self.continue_execution = False
    self.main_thread = None

    self.save_frames = save_frames
    if save_frames:
      assert save_dir_name is not None
      self.depth_logger = VisualLogger(
          duration=60,
          frequency=30, 
          data_save_path=os.path.join(save_dir_name, "depth")
      )
      self.depth_logger = VisualLogger(
          data_save_path=os.path.join(save_dir_name, "depth")
      )

  def start_thread(self):
    assert self.find_camera()
    self.main_thread = threading.Thread(target=self.streaming)
    self.continue_execution = True
    self.main_thread.start()

  def stop_thread(self):
    self.continue_execution = False
    self.main_thread.join()
    if self.save_frames:
      self.depth_logger.write()
    print("Realsense thread terminated")

  def find_camera(self):
    self.ctx = rs.context()
    self.device = self.ctx.query_devices()[0]
    self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
    print("Found: ", self.device_product_line)
    self.found_rgb = False
    self.found_depth = False
    for s in self.device.sensors:
      if s.get_info(rs.camera_info.name)=='RGB Camera':
          self.found_rgb = True
      if s.get_info(rs.camera_info.name)=='Stereo Module':
          self.found_depth = True
    assert self.found_depth or (not self.use_depth)
    assert self.found_rgb or (not self.use_rgb)
    return (self.found_depth or (not self.use_depth)) \
      and self.found_rgb or (not self.use_rgb)

  def streaming(self):
      self.config = rs.config()
      if self.use_rgb:
        self.config.enable_stream(
          rs.stream.color,
          RGB_WIDTH, RGB_HEIGHT,
          rs.format.bgr8, RGB_RATE
        )
      if self.use_depth:
        self.config.enable_stream(
          rs.stream.depth,
          DEPTH_WIDTH, DEPTH_HEIGHT,
          rs.format.z16,
          DEPTH_RATE
        )
      self.profile = self.pipeline.start(self.config)

      try:
        while self.continue_execution:
          frames = self.pipeline.wait_for_frames()
          if self.use_rgb:
            color_frame = frames.get_color_frame()
            assert color_frame
            # convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            self.rgb_frame = process_rgb(color_image, self.target_shape)

          if self.use_depth:
            depth_frame = frames.get_depth_frame()
            assert depth_frame
            # convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            self.depth_frame = process_depth(depth_image, self.target_shape)

          if self.save_frames:
            if self.use_depth:
              self.depth_array_saver.record(depth_image * self.get_depth_scale())
            if self.use_rgb:
              self.rgb_array_saver.record(color_frame)
      finally:
          self.pipeline.stop()

  def get_depth_frame(self):
    depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
    return depth_scale, self.depth_frame

  def get_rgb_frame(self):
    return self.rgb_frame
