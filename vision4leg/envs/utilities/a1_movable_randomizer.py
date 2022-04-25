"""Generates a random terrain at Minitaur gym environment reset."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from pybullet_envs.minitaur.envs import env_randomizer_base
import random
import numpy as np
import enum
import math
import itertools
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
  inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.dirname(parentdir))
os.sys.path.insert(0, parentdir)


_GRID_LENGTH = 15
_GRID_WIDTH = 2
_MAX_SAMPLE_SIZE = 30
_MIN_BLOCK_DISTANCE = 0.2
_MAX_BLOCK_LENGTH = _MIN_BLOCK_DISTANCE
_MIN_BLOCK_LENGTH = _MAX_BLOCK_LENGTH / 2
_MAX_BLOCK_HEIGHT = 0.075
_MIN_BLOCK_HEIGHT = _MAX_BLOCK_HEIGHT / 2


useProgrammatic = 0

heightfieldSource = useProgrammatic
numHeightfieldRows = 256
numHeightfieldColumns = 256
FLAG_TO_FILENAME = {
  'mounts': "heightmaps/wm_height_out.png",
  'maze': "heightmaps/Maze.png"
}

BOX_INIT_POSITION = {
  'random_mount': 0.85,
  'plane': 0,
  'random_hill': 1.58,
  'random_blocks': 0,
  'triangle_mesh': 1.98,
  'random_blocks_sparse': 0,
  'maze': 0,
  'simple_track': 0,
}

QUADRUPED_INIT_POSITION = {
  'random_mount': [[1, 1, 1.56], [1, 1, 1.76], [1, 1, 2.06], [1.5, 2, 2.56]],
  'plane': [-1, 0, 0.32],
  'random_hill': [0, 0, 2.25],
  'random_blocks': [-1, 0, 0.32],
  'triangle_mesh': [0, 0, 0.45],
  'random_blocks_sparse': [-3, 0, 0.32],
  'maze': [0, 0, 0.32],
  'random_heightfield': [0, 0, 0.32],
  'simple_track': [0, 0, 0.32],
  'sparse_with_height': [0, 0, 0.45],
  'stairs': [-0.15, 0, 0.32],
  'multi_stairs': [-0.3, 0, 0.32]
}

DIRECTION = [
  np.array([0.005, 0]),
  np.array([-0.005, 0]),
  np.array([0, 0.005]),
  np.array([0, -0.005]),
  np.array([0.004, 0.004]),
  np.array([-0.004, 0.004]),
  np.array([0.004, -0.004]),
  np.array([-0.004, -0.004]),
  np.array([0.002, 0.006]),
  np.array([-0.002, 0.006]),
  np.array([0.002, -0.006]),
  np.array([-0.002, -0.006]),
  np.array([0.006, 0.002]),
  np.array([-0.006, 0.002]),
  np.array([0.006, -0.002]),
  np.array([-0.006, -0.002]),
  np.array([0, 0]),
  np.array([0, 0]),
  np.array([0, 0]),
  np.array([0, 0]),
]


class PoissonDisc2D(object):
  """Generates 2D points using Poisson disk sampling method.

  Implements the algorithm described in:
    http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.mpdf
  Unlike the uniform sampling method that creates small clusters of points,
  Poisson disk method enforces the minimum distance between points and is more
  suitable for generating a spatial distribution of non-overlapping objects.
  """

  def __init__(self, grid_length, grid_width, min_radius, max_sample_size):
    """Initializes the algorithm.

    Args:
      grid_length: The length of the bounding square in which points are
        sampled.
      grid_width: The width of the bounding square in which points are
        sampled.
      min_radius: The minimum distance between any pair of points.
      max_sample_size: The maximum number of sample points around a active site.
        See details in the algorithm description.
    """
    self._cell_length = min_radius / math.sqrt(2)
    self._grid_length = grid_length
    self._grid_width = grid_width
    self._grid_size_x = int(grid_length / self._cell_length) + 1
    self._grid_size_y = int(grid_width / self._cell_length) + 1
    self._min_radius = min_radius
    self._max_sample_size = max_sample_size

    # Flattern the 2D grid as an 1D array. The grid is used for fast nearest
    # point searching.
    self._grid = [None] * self._grid_size_x * self._grid_size_y

    # Generate the first sample point and set it as an active site.
    first_sample = np.array(np.random.random_sample(
      2)) * [grid_length, grid_width]
    self._active_list = [first_sample]

    # Also store the sample point in the grid.
    self._grid[self._point_to_index_1d(first_sample)] = first_sample

  def _point_to_index_1d(self, point):
    """Computes the index of a point in the grid array.

    Args:
      point: A 2D point described by its coordinates (x, y).

    Returns:
      The index of the point within the self._grid array.
    """
    return self._index_2d_to_1d(self._point_to_index_2d(point))

  def _point_to_index_2d(self, point):
    """Computes the 2D index (aka cell ID) of a point in the grid.

    Args:
      point: A 2D point (list) described by its coordinates (x, y).

    Returns:
      x_index: The x index of the cell the point belongs to.
      y_index: The y index of the cell the point belongs to.
    """
    x_index = int(point[0] / self._cell_length)
    y_index = int(point[1] / self._cell_length)
    return x_index, y_index

  def _index_2d_to_1d(self, index2d):
    """Converts the 2D index to the 1D position in the grid array.

    Args:
      index2d: The 2D index of a point (aka the cell ID) in the grid.

    Returns:
      The 1D position of the cell within the self._grid array.
    """
    return index2d[0] + index2d[1] * self._grid_size_x

  def _is_in_grid(self, point):
    """Checks if the point is inside the grid boundary.

    Args:
      point: A 2D point (list) described by its coordinates (x, y).

    Returns:
      Whether the point is inside the grid.
    """
    return (0 <= point[0] < self._grid_length) and (0 <= point[1] < self._grid_width)

  def _is_in_range(self, index2d):
    """Checks if the cell ID is within the grid.

    Args:
      index2d: The 2D index of a point (aka the cell ID) in the grid.

    Returns:
      Whether the cell (2D index) is inside the grid.
    """

    return (0 <= index2d[0] < self._grid_size_x) and (0 <= index2d[1] < self._grid_size_y)

  def _is_close_to_existing_points(self, point):
    """Checks if the point is close to any already sampled (and stored) points.

    Args:
      point: A 2D point (list) described by its coordinates (x, y).

    Returns:
      True iff the distance of the point to any existing points is smaller than
      the min_radius
    """
    px, py = self._point_to_index_2d(point)
    # Now we can check nearby cells for existing points
    for neighbor_cell in itertools.product(range(px - 1, px + 2), range(py - 1, py + 2)):

      if not self._is_in_range(neighbor_cell):
        continue

      maybe_a_point = self._grid[self._index_2d_to_1d(neighbor_cell)]
      if maybe_a_point is not None and np.linalg.norm(maybe_a_point - point) < self._min_radius:
        return True

    return False

  def sample(self):
    """Samples new points around some existing point.

    Removes the sampling base point and also stores the new jksampled points if
    they are far enough from all existing points.
    """
    active_point = self._active_list.pop()
    for _ in range(self._max_sample_size):
      # Generate random points near the current active_point between the radius
      random_radius = np.random.uniform(
        self._min_radius, 2 * self._min_radius)
      random_angle = np.random.uniform(0, 2 * math.pi)

      # The sampled 2D points near the active point
      sample = random_radius * np.array([np.cos(random_angle),
                                         np.sin(random_angle)]) + active_point

      if not self._is_in_grid(sample):
        continue

      if self._is_close_to_existing_points(sample):
        continue

      self._active_list.append(sample)
      self._grid[self._point_to_index_1d(sample)] = sample

  def generate(self):
    """Generates the Poisson disc distribution of 2D points.

    Although the while loop looks scary, the algorithm is in fact O(N), where N
    is the number of cells within the grid. When we sample around a base point
    (in some base cell), new points will not be pushed into the base cell
    because of the minimum distance constraint. Once the current base point is
    removed, all future searches cannot start from within the same base cell.

    Returns:
      All sampled points. The points are inside the quare [0, grid_length] x [0,
      grid_width]
    """

    while self._active_list:
      self.sample()

    all_sites = []
    for p in self._grid:
      if p is not None:
        all_sites.append(p)

    return all_sites


class TerrainType(enum.Enum):
  """The randomzied terrain types we can use in the gym env."""
  PLANE = 0
  RANDOM_BLOCKS = 1
  TRIANGLE_MESH = 2
  RANDOM_HEIGHTFIELD = 3
  RANDOM_BLOCKS_SPARSE = 4
  RANDOM_HILL = 5
  RANDOM_MOUNT = 6
  MAZE = 7
  SIMPLE_TRACK = 8
  GOAL_MOUNT = 10


terrain_dict = {
  'plane': TerrainType.PLANE,
  'random_blocks': TerrainType.RANDOM_BLOCKS,
  'triangle_mesh': TerrainType.TRIANGLE_MESH,
  'random_heightfield': TerrainType.RANDOM_HEIGHTFIELD,
  'random_blocks_sparse': TerrainType.RANDOM_BLOCKS_SPARSE,
  'random_hill': TerrainType.RANDOM_HILL,
  'random_mount': TerrainType.RANDOM_MOUNT,
  'maze': TerrainType.MAZE,
  'simple_track': TerrainType.SIMPLE_TRACK,
}


class TerrainRandomizer(env_randomizer_base.EnvRandomizerBase):
  """Generates an uneven terrain in the gym env."""

  def __init__(self,
               terrain_type=TerrainType.RANDOM_HEIGHTFIELD,
               mesh_filename="robotics/reinforcement_learning/minitaur/envs/testdata/"
               "triangle_mesh_terrain/terrain9735.obj",
               height_range=0.05,
               prob=0.1,
               sparse=False,
               simple=False,
               single=False,
               multiple=False,
               mesh_scale=None,
               dynamic=False,
               total_randomize=False
               ):
    """Initializes the randomizer.

    Args:
      terrain_type: Whether to generate random blocks or load a triangle mesh.
      mesh_filename: The mesh file to be used. The mesh will only be loaded if
        terrain_type is set to TerrainType.TRIANGLE_MESH.
      mesh_scale: the scaling factor for the triangles in the mesh file.
    """
    self._terrain_type = terrain_type
    self._mesh_filename = mesh_filename
    self._mesh_scale = mesh_scale if mesh_scale else [0.6, 0.3, 0.05]
    self.height_range = height_range
    self.sparse = sparse
    self.simple = simple
    self.single = single
    self.multiple = multiple
    self.dynamic = dynamic
    self.total_randomize = total_randomize
    self.prob = prob
    if self._terrain_type == TerrainType.TRIANGLE_MESH:
      file_path = os.path.join(os.path.dirname(
        __file__), '../../assets', self._mesh_filename)
      with open(file_path, 'r') as in_f:
        self.v_lines = []
        self.f_lines = []
        # lines = []
        for line in in_f.readlines():
          # print(line.split())
          items = line.split()
          if items[0] == 'v':
            line = ['v'] + [float(i) for i in line.split()[1:]]
            self.v_lines.append(line)
          elif items[0] == 'f':
            self.f_lines.append(line)

    self.heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns

  def randomize_env(self, env):
    self.step_from_reset = 0
    if self._terrain_type is TerrainType.TRIANGLE_MESH:
      self._load_triangle_mesh(env)
    elif self._terrain_type is TerrainType.RANDOM_BLOCKS:
      self._generate_convex_blocks(env)
    elif self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE:
      self._generate_convex_blocks_sparse(env)
    elif self._terrain_type is TerrainType.RANDOM_HEIGHTFIELD:
      self._generate_field(env)
    elif self._terrain_type is TerrainType.RANDOM_HILL:
      self._generate_terrain(env)
    elif self._terrain_type is TerrainType.RANDOM_MOUNT:
      self._generate_terrain(env)
    elif self._terrain_type is TerrainType.MAZE:
      self._generate_terrain(env)
    elif self._terrain_type is TerrainType.SIMPLE_TRACK:
      self._generate_simple_track(env)
    else:
      raise NotImplementedError

  def randomize_step(self, env):
    if not self.dynamic:
      return
    self.step_from_reset += 1
    if self._terrain_type is TerrainType.TRIANGLE_MESH:
      pass
    elif self._terrain_type is TerrainType.RANDOM_BLOCKS:
      self._randomize_random_convex_blocks(env)
    elif self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE:
      self._randomize_random_blocks_sparse(env)
    elif self._terrain_type is TerrainType.RANDOM_HEIGHTFIELD:
      pass
    elif self._terrain_type is TerrainType.RANDOM_HILL:
      pass
    elif self._terrain_type is TerrainType.RANDOM_MOUNT:
      pass
    elif self._terrain_type is TerrainType.MAZE:
      pass
    elif self._terrain_type is TerrainType.SIMPLE_TRACK:
      self._randomize_simple_track(env)
    else:
      raise NotImplementedError

    self.update_randomize_direction(env)

  def randomize_reset(self, env):
    if not self.dynamic:
      return
    self.step_from_reset = 0
    if self._terrain_type is TerrainType.TRIANGLE_MESH:
      pass
    elif self._terrain_type is TerrainType.RANDOM_BLOCKS:
      self._reset_random_convex_blocks(env)
    elif self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE:
      self._reset_random_blocks_sparse(env)
    elif self._terrain_type is TerrainType.RANDOM_HEIGHTFIELD:
      pass
    elif self._terrain_type is TerrainType.RANDOM_HILL:
      pass
    elif self._terrain_type is TerrainType.RANDOM_MOUNT:
      pass
    elif self._terrain_type is TerrainType.MAZE:
      pass
    elif self._terrain_type is TerrainType.SIMPLE_TRACK:
      self._reset_simple_track(env)
    else:
      raise NotImplementedError

    self.update_randomize_direction(env)

  def update_randomize_direction(self, env):
    if self._terrain_type not in [TerrainType.RANDOM_BLOCKS_SPARSE, TerrainType.SIMPLE_TRACK, TerrainType.RANDOM_BLOCKS]:
      return

    if self.simple:
      if self.step_from_reset % 120 == 0:
        if not self.total_randomize:
          for i in range(len(self.block_randomized_direction)):
            if self.block_randomized_direction[i] == 0:
              self.block_randomized_direction[i] = 1
            elif self.block_randomized_direction[i] == 1:
              self.block_randomized_direction[i] = 0
            elif self.block_randomized_direction[i] == 2:
              self.block_randomized_direction[i] = 3
            elif self.block_randomized_direction[i] == 3:
              self.block_randomized_direction[i] = 2
        else:
          self.block_randomized_direction = np.random.randint(
            0, 20, size=(150,))
    else:
      if self.step_from_reset % 200 == 0:
        for i in range(len(self.block_randomized_direction)):
          if self.block_randomized_direction[i] == 0:
            self.block_randomized_direction[i] = 1
          elif self.block_randomized_direction[i] == 1:
            self.block_randomized_direction[i] = 0
          elif self.block_randomized_direction[i] == 2:
            self.block_randomized_direction[i] = 3
          elif self.block_randomized_direction[i] == 3:
            self.block_randomized_direction[i] = 2

  def _load_triangle_mesh(self, env):
    """Represents the random terrain using a triangle mesh.

    It is possible for Minitaur leg to stuck at the common edge of two triangle
    pieces. To prevent this from happening, we recommend using hard contacts
    (or high stiffness values) for Minitaur foot in sim.

    Args:
      env: A minitaur gym environment.
    """
    env.pybullet_client.removeBody(env._world_dict["ground"])

    rand = np.random.rand()
    temp_file_name = "/tmp/temp_generated_terrain_{}.obj".format(rand)
    with open(temp_file_name, 'w') as out_f:
      post_lines = []
      for line in self.v_lines:
        height = line[3] + (np.random.rand() * 1)
        if np.abs(line[1]) < 0.5 and np.abs(line[2]) < 0.5:
          height = line[3]
        if np.abs(line[1]) < 1 and np.abs(line[2]) < 1:
          height = line[3] + (np.random.rand() * 0.25)

        post_lines.append(
          "{} {} {} {}\n".format(
            line[0], line[1], line[2], height
          )
        )

      out_f.writelines(
        ['o Terrain\n'] + post_lines + self.f_lines
      )

    terrain_collision_shape_id = env.pybullet_client.createCollisionShape(
      shapeType=env.pybullet_client.GEOM_MESH,
      # fileName=self._mesh_filename,
      fileName=temp_file_name,
      flags=1,
      meshScale=self._mesh_scale)
    terrain_visual_shape_id = env.pybullet_client.createVisualShape(
      shapeType=env.pybullet_client.GEOM_MESH,
      # fileName=self._mesh_filename,
      fileName=temp_file_name,
      flags=1,
      meshScale=self._mesh_scale,
      rgbaColor=(0.3, 0.3, 0.3)
    )
    env._world_dict["ground"] = env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=terrain_collision_shape_id,
      baseVisualShapeIndex=terrain_visual_shape_id,
      basePosition=[0, 0, 0])

  def _generate_convex_blocks(self, env):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """
    block_centers = np.split(np.random.uniform(
      [-2, -3], [5, 3], size=(150, 2)), 150)
    self.block_centers = block_centers
    blocks_list = []
    for center in block_centers:
      # We want the blocks to be in front of the robot.
      # shifted_center = np.array(center) - [2, _GRID_WIDTH / 2]
      shifted_center = center.reshape(2)
      # Do not place blocks near the point [0, 0], where the robot will start.
      if abs(shifted_center[0]) < 0.3 and abs(shifted_center[1]) < 0.3:
        continue
      half_length = np.random.uniform(
        _MIN_BLOCK_LENGTH, _MAX_BLOCK_LENGTH) / (2 * math.sqrt(2))
      half_height = np.random.uniform(
        _MIN_BLOCK_HEIGHT, _MAX_BLOCK_HEIGHT) / 2
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[half_length, half_length, half_height])

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[
          half_length, half_length, half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1))

      total_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[shifted_center[0], shifted_center[1], half_height])
      blocks_list.append(total_id)

    env._world_dict["blocks_list"] = blocks_list
    self.block_randomized_direction = np.random.randint(0, 4, size=(150,))

  def _randomize_random_convex_blocks(self, env):
    blocks_list = env._world_dict["blocks_list"]

    for block, direction_id in zip(blocks_list, self.block_randomized_direction):
      pos, ori = env.pybullet_client.getBasePositionAndOrientation(block)
      pos = list(pos)
      pos[0] += DIRECTION[direction_id][0]
      pos[1] += DIRECTION[direction_id][1]
      env.pybullet_client.resetBasePositionAndOrientation(block,
                                                          posObj=pos,
                                                          ornObj=ori)

  def _reset_random_convex_blocks(self, env):
    blocks_list = env._world_dict["blocks_list"]

    for block, init_pos in zip(blocks_list, self.block_centers):
      pos, ori = env.pybullet_client.getBasePositionAndOrientation(block)
      pos = list(pos)
      pos[0] = init_pos[0]
      pos[1] = init_pos[1]
      env.pybullet_client.resetBasePositionAndOrientation(block,
                                                          posObj=pos,
                                                          ornObj=ori)

  def _generate_simple_track(self, env):
    half_length = 0.5
    half_height = 0.5
    if self.dynamic:
      block_centers = [
        np.array([0.75, -0.3]),
        np.array([2.5, -1.5]),
        np.array([2.5, 1.5]),

      ]
    else:
      block_centers = [
        np.array([0.75, 0.]),
        np.array([2.5, -1.5]),
        np.array([2.5, 1.5]),

      ]
    self.block_centers = block_centers
    self.blocks_list = []
    for center in block_centers:
      # We want the blocks to be in front of the robot.
      # shifted_center = np.array(center) - [2, _GRID_WIDTH / 2]
      shifted_center = center.reshape(2)
      # print(center)
      # Do not place blocks near the point [0, 0], where the robot will start.
      if abs(shifted_center[0]) < 0.5 and abs(shifted_center[1]) < 0.3 and not self.simple:
        continue
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[half_length + 0.035, half_length + 0.035, half_height])

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[
          half_length, half_length, half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1))

      total_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[shifted_center[0], shifted_center[1], half_height])
      self.blocks_list.append(total_id)
    self.block_randomized_direction = [2, 1, 1]

    # Fench
    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        10, half_length, 3 * half_height
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[10, half_length, 3 * half_height],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[5, 2.3, 3 * half_height]
    )

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        10, half_length, 3 * half_height
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[10, half_length, 3 * half_height],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[5, -2.3, 3 * half_height]
    )

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        half_length, 10, 3 * half_height
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[half_length, 10, 3 * half_height],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[-6, 0, 3 * half_height]
    )

  def _randomize_simple_track(self, env):
    for block, direction_id in zip(self.blocks_list, self.block_randomized_direction):
      pos, ori = env.pybullet_client.getBasePositionAndOrientation(block)
      pos = list(pos)
      pos[0] += DIRECTION[direction_id][0]
      pos[1] += DIRECTION[direction_id][1]
      env.pybullet_client.resetBasePositionAndOrientation(block,
                                                          posObj=pos,
                                                          ornObj=ori)

  def _reset_simple_track(self, env):
    for block, init_pos in zip(self.blocks_list, self.block_centers):
      pos, ori = env.pybullet_client.getBasePositionAndOrientation(block)
      pos = list(pos)
      pos[0] = init_pos[0]
      pos[1] = init_pos[1]
      env.pybullet_client.resetBasePositionAndOrientation(block,
                                                          posObj=pos,
                                                          ornObj=ori)

  def _generate_convex_blocks_sparse(self, env):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """
    half_length = 0.5 / (2 * math.sqrt(2))
    half_height = 1
    if self.simple:
      half_length = 0.2
      # half_height = np.random.uniform(_MIN_BLOCK_HEIGHT, _MAX_BLOCK_HEIGHT) / 2
      half_height = 2
      if self.multiple:
        half_height = 0.5

    if self.simple:
      # block_centers = np.split(
      #   np.random.uniform([0.1, -2.5], [15, 2.5], size=(40,2)), 3)
      if self.single:
        block_centers = [
          np. array([0.75, 0]),
        ]
      elif self.multiple:
        block_centers = []
        for i in range(20):
          if i % 2 == 0:
            temp_centers = np.array([
              [1, 0],
              [1, 1.5],
              [1, -1.5],
              [1, -3.],
              [1, 3.],
              [1, -4.5],
              [1, 4.5]]
            )
          else:
            temp_centers = np.array([
              [1, 0.75],
              [1, -0.75],
              [1, -2.25],
              [1, 2.25],
              [1, -3.5],
              [1, 3.5]
            ])
          # temp_centers[:, 0] = i * 0.5 + 0.5
          temp_centers[:, 0] = 2.0 * i + 0.5
          temp_centers[:, 1] *= 1.2
          # print(temp_centers.shape)
          block_centers += np.split(temp_centers,
                                    temp_centers.shape[0])
      else:
        block_centers = [
          np.array([1, 0]),
          np.array([1, 0.7]),
          np.array([1, -0.7]),
          np.array([1, -1.4]),
          np.array([1, 1.4]),
        ]
    else:
      # block_centers = np.split(np.random.uniform(
      #     [-5, -10], [15, 10], size=(100, 2)), 100)
      if self.single:
        block_centers = [
          np. array([0.75, 0]),
        ]
      elif self.multiple:
        block_centers = []
        for i in range(50):
          if i % 2 == 0:
            temp_centers = np.array(
              [[1, 3]],
            )
          else:
            temp_centers = np.array(
              [[1, -3]],
            )
          # temp_centers[:, 0] = i * 0.5 + 0.5
          temp_centers[:, 0] = i + 0.5
          # temp_centers[:, 1] *= 1.2
          # print(temp_centers.shape)
          block_centers += np.split(temp_centers,
                                    temp_centers.shape[0])
      else:
        block_centers = [
          np.array([1, 0]),
          np.array([1, 0.7]),
          np.array([1, -0.7]),
          np.array([1, -1.4]),
          np.array([1, 1.4]),
        ]

    # create_time = time.time()
    self.block_centers = block_centers
    self.blocks_list = []
    for center in block_centers:
      # We want the blocks to be in front of the robot.
      # shifted_center = np.array(center) - [2, _GRID_WIDTH / 2]
      shifted_center = center.reshape(2)
      # print(center)
      # Do not place blocks near the point [0, 0], where the robot will start.
      if abs(shifted_center[0]) < 0.5 and abs(shifted_center[1]) < 0.3 and not self.simple:
        continue
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[half_length + 0.035, half_length + 0.035, half_height])

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[
          half_length, half_length, half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1))

      total_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[shifted_center[0], shifted_center[1], half_height])

      self.blocks_list.append(total_id)
    if self.simple:
      self.block_randomized_direction = np.random.randint(
        0, 4, size=(len(self.blocks_list,)))
    else:
      self.block_randomized_direction = np.array(
        [2 if i % 2 == 0 else 3 for i in range(len(self.blocks_list))])
    # Fench
    if self.simple:
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          10 + 0.035, half_length + 0.035, 3 * half_height
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[10, half_length, 3 * half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[5, 6.5, 3 * half_height]
      )

      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          10 + 0.035, half_length + 0.035, 3 * half_height
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[10, half_length, 3 * half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[5, -6.5, 3 * half_height]
      )
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          half_length, 10, 3 * half_height
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[half_length, 10, 3 * half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[-6, 0, 3 * half_height]
      )
    else:
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          10 + 0.035, half_length + 0.035, 3 * half_height
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[10, half_length, 3 * half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[5, 4, 3 * half_height]
      )

      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          10 + 0.035, half_length + 0.035, 3 * half_height
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[10, half_length, 3 * half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[5, -4, 3 * half_height]
      )
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          half_length, 10, 3 * half_height
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[half_length, 10, 3 * half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[-6, 0, 3 * half_height]
      )

  def _randomize_random_blocks_sparse(self, env):
    scale = 2 if self.simple else 5
    for block, direction_id in zip(self.blocks_list, self.block_randomized_direction):
      pos, ori = env.pybullet_client.getBasePositionAndOrientation(block)
      pos = list(pos)
      pos[0] += DIRECTION[direction_id][0] * scale
      pos[1] += DIRECTION[direction_id][1] * scale
      env.pybullet_client.resetBasePositionAndOrientation(block,
                                                          posObj=pos,
                                                          ornObj=ori)

  def _reset_random_blocks_sparse(self, env):
    for block, init_pos in zip(self.blocks_list, self.block_centers):
      pos, ori = env.pybullet_client.getBasePositionAndOrientation(block)
      pos = list(pos)
      pos[0] = init_pos[0][0]
      pos[1] = init_pos[0][1]
      env.pybullet_client.resetBasePositionAndOrientation(block,
                                                          posObj=pos,
                                                          ornObj=ori)

  def _generate_field(self, env):
    heightPerturbationRange = self.height_range
    heightPerturbationRange = heightPerturbationRange

    for j in range(int(numHeightfieldColumns / 2)):
      for i in range(int(numHeightfieldRows / 2)):
        if self.sparse:
          if np.random.rand() < self.prob:
            height = 0
          height = heightPerturbationRange
        else:
          height = random.uniform(0, heightPerturbationRange)
        self.heightfieldData[2 * i +
                             2 * j * numHeightfieldRows] = height
        self.heightfieldData[2 * i + 1 +
                             2 * j * numHeightfieldRows] = height
        self.heightfieldData[2 * i + (2 * j + 1) *
                             numHeightfieldRows] = height
        self.heightfieldData[2 * i + 1 + (2 * j + 1) *
                             numHeightfieldRows] = height
    for i in range(-2, 2):
      for j in range(-2, 2):
        x = int(numHeightfieldColumns / 4) + i
        y = int(numHeightfieldRows / 4) + j
        self.heightfieldData[2 * x +
                             2 * y * numHeightfieldRows] = 0
        self.heightfieldData[2 * x + 1 +
                             2 * y * numHeightfieldRows] = 0
        self.heightfieldData[2 * x + (2 * y + 1) *
                             numHeightfieldRows] = 0
        self.heightfieldData[2 * x + 1 + (2 * y + 1) *
                             numHeightfieldRows] = 0

    terrainShape = env.pybullet_client.createCollisionShape(
      shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
      meshScale=[.07, .07, 1.6],
      heightfieldTextureScaling=0,
      heightfieldData=self.heightfieldData,
      numHeightfieldRows=numHeightfieldRows,
      numHeightfieldColumns=numHeightfieldColumns)
    terrain = env.pybullet_client.createMultiBody(0, terrainShape)
    env.pybullet_client.resetBasePositionAndOrientation(
      terrain, [0, 0, 0.0], [0, 0, 0, 1])

    env.pybullet_client.changeVisualShape(terrain,
                                          -1,
                                          rgbaColor=[0.5, 0.5, 0.5, 1],
                                          specularColor=[0.5, 0.5, 0.5, 1]
                                          )
    env._world_dict["terrain"] = terrain

  def _generate_terrain(self, env, height_perturbation_range=0.05):
    env.pybullet_client.setAdditionalSearchPath(mpd.getDataPath())
    height_perturbation_range = height_perturbation_range

    if self._terrain_type == TerrainType.RANDOM_HILL:
      terrain_shape = env.pybullet_client.createCollisionShape(
        shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
        meshScale=[.5, .5, .5],
        flags=1,
        fileName="heightmaps/ground0.txt",
        heightfieldTextureScaling=128)
      terrain = env.pybullet_client.createMultiBody(0, terrain_shape)
      textureId = env.pybullet_client.loadTexture(
        f"{mpd.getDataPath()}/grass.png")
      env.pybullet_client.changeVisualShape(
        terrain, -1, textureUniqueId=textureId, flags=1)
      env.pybullet_client.resetBasePositionAndOrientation(
        terrain, [1, 0, 2], [0, 0, 0, 1])

    elif self._terrain_type == TerrainType.RANDOM_MOUNT:
      terrain_shape = env.pybullet_client.createCollisionShape(
        shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
        meshScale=[0.1, 0.1, 15],
        flags=1,
        fileName=FLAG_TO_FILENAME["mounts"])
      terrain = env.pybullet_client.createMultiBody(0, terrain_shape)
      textureId = env.pybullet_client.loadTexture(
        "heightmaps/gimp_overlay_out.png")
      env.pybullet_client.changeVisualShape(
        terrain, -1, textureUniqueId=textureId, flags=1)
      env.pybullet_client.resetBasePositionAndOrientation(
        terrain, [0, 0, 2], [0, 0, 0, 1])

    elif self._terrain_type == TerrainType.MAZE:
      terrain_shape = env.pybullet_client.createCollisionShape(
        shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
        meshScale=[.3, .3, 2],
        fileName=FLAG_TO_FILENAME["maze"])
      terrain = env.pybullet_client.createMultiBody(0, terrain_shape)
      textureId = env.pybullet_client.loadTexture(
        "heightmaps/Maze.png")
      env.pybullet_client.changeVisualShape(
        terrain, -1, textureUniqueId=textureId)
      env.pybullet_client.resetBasePositionAndOrientation(
        terrain, [0, 0, 0], [0, 0, 0, 1])

    else:
      raise NotImplementedError

    self.terrain_shape = terrain_shape
    env.pybullet_client.changeVisualShape(
      terrain, -1, rgbaColor=[1, 1, 1, 1])
    env.pybullet_client.changeDynamics(terrain, -1, lateralFriction=50,
                                       spinningFriction=0.1,
                                       rollingFriction=0.1)
    env._world_dict["terrain"] = terrain
