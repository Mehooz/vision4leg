import numpy as np


class NormedStateHistory():
  def __init__(self,input_dim, num_hist, mean, var):
    '''
    Initialize the historical and normalization processor
    This class takes single sensor input, record them and scale according
    to mean and covariance stored in training.
    Input:
        input_dim - input dimension of a single sensor read
        num_hist - number of historical data saved for this sensor.
        mean - array storing mean that is computed in training. Size of this array should be (imput_dim * num_hist,).
        std - array storing standard deviation that is computed in training. Size of this array should be (imput_dim * num_hist,). 
    '''

    self.input_dim = input_dim
    self.num_hist = num_hist

    self.sensor_history = np.zeros((self.num_hist, self.input_dim))

    assert mean.shape[0] == self.input_dim * self.num_hist
    assert var.shape[0] == self.input_dim * self.num_hist

    self.mean = mean
    self.std = np.sqrt(var)

  def record_and_normalize(self, sensor_input, mean=None, std=None, backwards=False):
    '''
    record current sensor input in historical buffer
    output the normalized historical information.
    Input:
        sensor_input - sensor input to be inserted.
    
    Output:
        normalized historical data
    '''

    assert sensor_input.shape[0] == self.input_dim

    if mean is None:
      mean = self.mean
    if std is None:
      std = self.std

    # update history
    if backwards is True:
      self.sensor_history = np.vstack((self.sensor_history[1:,:], sensor_input))
    else: 
      self.sensor_history = np.vstack((sensor_input, self.sensor_history[:-1,:]))

    # normalize
    return (self.sensor_history.reshape(-1) - mean) / (std + 1e-4)

def depth_process(depth):
  depth[depth < 1e-3] = +np.inf
  depth = np.clip(depth, a_min=0.3, a_max=3)
  depth = np.sqrt(np.log(depth+1))
  return depth

class VisualHistory():
  def __init__(
    self,
    frame_shape,
    num_hist,
    mean, var,
    sliding_frames=True
  ):
    self.frame_shape = frame_shape
    self.num_hist = num_hist
    self.unrolled_frame_history = NormedStateHistory(
      np.prod(self.frame_shape), self.num_hist, mean, var
    )
    self.sliding_frames = sliding_frames

  def record_and_normalize(self, frame, depth_scale, backwards=False):
    depth = frame * depth_scale
    depth = depth_process(depth)
    unrolled_depth_history = self.unrolled_frame_history.record_and_normalize(
      depth.reshape(-1), backwards=backwards
    )
    if self.sliding_frames:
      # reshape back to (64,64,num_hist); where the two 64 comes from self.frame shape
      # Return only 0,4,8,12 index but now it is flattened, so
      # it will be indices 64*64*0:64*64*1, 64*64*4:64*64*5, 64*64*8:64*64*9, 64*64*12:
      unrolled_depth_history = np.array((
        unrolled_depth_history[64*64*0:64*64*1],
        unrolled_depth_history[64*64*4:64*64*5],
        unrolled_depth_history[64*64*8:64*64*9],
        unrolled_depth_history[64*64*12:])
      ).flatten()
    return unrolled_depth_history
