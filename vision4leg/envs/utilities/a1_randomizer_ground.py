"""Generates a random terrain at Minitaur gym environment reset."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random
from pybullet_envs.minitaur.envs import env_randomizer_base
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


FLAG_TO_FILENAME = {
  'mounts': "heightmaps/wm_height_out.png",
  'maze': "heightmaps/Maze.png"
}
GOAL_POS = {
  'mounts': [[4, 11.5, 3.5], [2.5, 8.0, 2.7], [2., 8.0, 2.7], [2., 7.5, 2.5]],
  'stairs': [10, 0, 0],
  'multi_stairs': [20, 0, 0],
}

_GRID_LENGTH = 15
_GRID_WIDTH = 2
_MAX_SAMPLE_SIZE = 30
numHeightfieldRows = 256
numHeightfieldColumns = 256
_MIN_BLOCK_DISTANCE = 0.2
_MAX_BLOCK_LENGTH = _MIN_BLOCK_DISTANCE
_MIN_BLOCK_LENGTH = _MAX_BLOCK_LENGTH / 2
_MAX_BLOCK_HEIGHT = 0.075
_MIN_BLOCK_HEIGHT = _MAX_BLOCK_HEIGHT / 2
_MAX_BLOCK_HEIGwm_height_outldRows = 256
numHeightfieldColumns = 256
SUBGOAL_POS = [(5, 3, 2), (12, 7, 2.3)]
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
    http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
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
  STAIRS = 8
  RANDOM_BLOCKS_SPARSE_AND_HEIGHTFIELD = 9
  GOAL_MOUNT = 10
  RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL = 11
  RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL_HEIGHTFIELD = 12
  RANDOM_SPHERE_WITH_SUBGOAL = 13
  MULTI_STAIRS = 14
  RANDOM_BLOCKS_SPARSE_THIN_WIDE = 15
  RANDOM_CHAIR_DESK = 16


TerrainTypeDict = {
  "plane": TerrainType.PLANE,
  "random_blocks": TerrainType.RANDOM_BLOCKS,
  "triangle_mesh": TerrainType.TRIANGLE_MESH,
  "random_heightfield": TerrainType.RANDOM_HEIGHTFIELD,
  "random_blocks_sparse": TerrainType.RANDOM_BLOCKS_SPARSE,
  "random_hill": TerrainType.RANDOM_HILL,
  "random_mount": TerrainType.RANDOM_MOUNT,
  "random_maze": TerrainType.MAZE,
  "stairs": TerrainType.STAIRS,
  "random_blocks_sparse_and_heightfield": TerrainType.RANDOM_BLOCKS_SPARSE_AND_HEIGHTFIELD,
  "mount": TerrainType.GOAL_MOUNT,
  "random_blocks_sparse_with_subgoal": TerrainType.RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL,
  "random_blocks_sparse_with_subgoal_heightfield": TerrainType.RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL_HEIGHTFIELD,
  "random_sphere_with_subgoal": TerrainType.RANDOM_SPHERE_WITH_SUBGOAL,
  "multi_stairs": TerrainType.MULTI_STAIRS,
  "random_blocks_sparse_thin_wide": TerrainType.RANDOM_BLOCKS_SPARSE_THIN_WIDE,
  "random_chair_desk": TerrainType.RANDOM_CHAIR_DESK
}


QUADRUPED_INIT_POSITION = {
  'random_mount': [[1, 1, 1.56], [1, 1, 1.76], [2., 3.3, 2.26], [2., 3.3, 2.76]],
  'mount': [1, 1, 1.56],
  'plane': [0, 0, 0.32],
  'random_hill': [0, 0, 2.25],
  'random_blocks': [0, 0, 0.32],
  'triangle_mesh': [0, 0, 0.45],
  'random_blocks_sparse': [0, 0, 0.32],
  'random_heightfield': [0, 0, 0.32],
  'simple_track': [0, 0, 0.32],
  'random_maze': [0, 0, 0.32],
  'stairs': [-0.15, 0, 0.32],
  'multi_stairs': [1.0, 0, 0.42],
  'random_chair_desk': [0, 0, 0.32],
  'random_blocks_sparse_and_heightfield': [0, 0, 0.32],
  'random_blocks_sparse_with_subgoal_heightfield': [0, 0, 0.32],
  'random_blocks_sparse_with_subgoal': [0, 0, 0.32],
  'random_blocks_sparse_thin_wide': [0, 0, 0.32],
  'random_sphere_with_subgoal': [0, 0, 0.32],
}
QUADRUPED_INIT_ORI = {
  'random_mount': [[0, 0, 0.6, 1], [0, 0, 0.4, 1], [0, 0, 2.0, 1], [0, 0, 2.0, 1]],
  'mount': [0, 0, 0.8, 1],
}
MOUNT_LEVEL = [1, 1., 1., 1.]
# MOUNT_HEIGHT = [1, 2, 5, 10]


class TerrainRandomizer(env_randomizer_base.EnvRandomizerBase):
  """Generates an uneven terrain in the gym env."""

  def __init__(self,
               terrain_type=TerrainType.RANDOM_HEIGHTFIELD,
               mesh_filename="robotics/reinforcement_learning/minitaur/envs/testdata/"
               "triangle_mesh_terrain/terrain9750.obj",
               height_range=0.05,
               mesh_scale=None,
               random_shape=False,
               moving=False
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
    self.mount_level = 0
    self.moving = moving
    self.block_randomized_direction = np.random.randint(0, 20, size=(150,))
    self.prob = 0.4
    if self._terrain_type == TerrainType.TRIANGLE_MESH:
      # self.pybullet_client.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), '../assets'))
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
    self.box_ids = []
    self.triangles = []
    self._created = False
    self.random_shape = random_shape
    self.terrain_created = False
    self.goal = [terrain_type in [TerrainType.GOAL_MOUNT,
                                  TerrainType.STAIRS, TerrainType.MULTI_STAIRS]]
    self.subgoal = (terrain_type in [TerrainType.RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL,
                                     TerrainType.RANDOM_SPHERE_WITH_SUBGOAL, TerrainType.RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL_HEIGHTFIELD])

  def randomize_env(self, env):
    """Generate a random terrain for the current env.

    Args:
      env: A minitaur gym environment.
    """
    if self._terrain_type is TerrainType.TRIANGLE_MESH:
      self._load_triangle_mesh(env)
    if self._terrain_type is TerrainType.RANDOM_BLOCKS:
      self._generate_convex_blocks(env)
    if self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE:
      self._generate_convex_blocks_sparse(env)
    if self._terrain_type is TerrainType.RANDOM_HEIGHTFIELD:
      self._generate_field(env)
    if self._terrain_type is TerrainType.STAIRS:
      self._generate_stairs(env)
    if self._terrain_type is TerrainType.MULTI_STAIRS:
      self._generate_multi_stairs(env)
    if self._terrain_type in [
        TerrainType.RANDOM_HILL,
        TerrainType.RANDOM_MOUNT,
        TerrainType.MAZE,
        TerrainType.GOAL_MOUNT,
    ] and not self.terrain_created:
      self._generate_terrain(env)
    if self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE_AND_HEIGHTFIELD:
      self._generate_convex_blocks_sparse(env)
      self._generate_field(env, num_rows=512, num_columns=64)
    if self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL:
      self._generate_convex_blocks_sparse_hard_with_subgoal(env)
    if self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL_HEIGHTFIELD:
      self._generate_convex_blocks_sparse_hard_with_subgoal(env)
      self._generate_field(env, num_rows=512, num_columns=64)
    if self._terrain_type is TerrainType.RANDOM_SPHERE_WITH_SUBGOAL:
      self._generate_spheres_and_subgoal(env)
    if self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE_THIN_WIDE:
      self._generate_convex_blocks_thin_wide(
        env)
    if self._terrain_type is TerrainType.RANDOM_CHAIR_DESK:
      self._generate_chair_desk(env, with_subgoal=self.subgoal)
    if self._terrain_type is TerrainType.MAZE:
      self._sample_goal_in_maze(env)

  def randomize_step(self, env):
    if not self.moving:
      return
    if self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE_THIN_WIDE:
      self._randomize_random_blocks_sparse(env)
    elif self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE_AND_HEIGHTFIELD:
      self._randomize_random_blocks_sparse(env)
    elif self._terrain_type is TerrainType.RANDOM_BLOCKS_SPARSE:
      self._randomize_random_blocks_sparse(env)
    else:
      pass

    self.update_randomize_direction(env)

  def update_randomize_direction(self, env):
    if self._terrain_type not in [TerrainType.RANDOM_BLOCKS_SPARSE, TerrainType.RANDOM_BLOCKS_SPARSE_THIN_WIDE, TerrainType.RANDOM_BLOCKS_SPARSE_AND_HEIGHTFIELD]:
      return

    if env._env_step_counter % 150 == 0:
      # self.block_randomized_direction = np.random.randint(
      #     0, 20, size=(70,))
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
          self.block_randomized_direction[i] = np.random.randint(0, 20)

  def _load_triangle_mesh(self, env):
    """Represents the random terrain using a triangle mesh.

    It is possible for Minitaur leg to stuck at the common edge of two triangle
    pieces. To prevent this from happening, we recommend using hard contacts
    (or high stiffness values) for Minitaur foot in sim.

    Args:
      env: A minitaur gym environment.
    """
    if self.terrain_created:
      return

    env.pybullet_client.removeBody(env.ground_id)

    mesh_scale = [0.2, 0.2, np.random.uniform(0.8, 1.)]
    terrain_collision_shape_id = env.pybullet_client.createCollisionShape(
      shapeType=env.pybullet_client.GEOM_MESH,
      fileName=self._mesh_filename,
      flags=1,
      meshScale=mesh_scale)
    env.ground_id = env.pybullet_client.createMultiBody(
      baseMass=0, baseCollisionShapeIndex=terrain_collision_shape_id, basePosition=[1.5, 0, 0])
    self.terrain_created = True

  def _generate_convex_blocks(self, env):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """
    block_centers = np.split(np.random.uniform(
      [0, -0.5], [5, 0.5], size=(20, 2)), 20)

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

      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[shifted_center[0], shifted_center[1], half_height])

  def _randomize_random_blocks_sparse(self, env):
    scale = 3
    for box_id, direction_id in zip(self.box_ids, self.block_randomized_direction):
      _, b_id = box_id
      pos, ori = env.pybullet_client.getBasePositionAndOrientation(b_id)
      pos = list(pos)
      pos[0] += DIRECTION[direction_id][0] * scale
      pos[1] += DIRECTION[direction_id][1] * scale
      env.pybullet_client.resetBasePositionAndOrientation(b_id,
                                                          posObj=pos,
                                                          ornObj=ori)

  def _move_block_pos(self, env):
    self.poisson_disc = PoissonDisc2D(26, 6, 1., 150)
    block_centers = self.poisson_disc.generate()
    np.random.shuffle(block_centers)
    for idx, box_id in enumerate(self.box_ids):
      _, b_id = box_id
      # x, y = pos
      shifted_center = block_centers[idx].reshape(
        2) + np.array([2.5, -3.0])
      env.pybullet_client.resetBasePositionAndOrientation(
        b_id,
        posObj=[shifted_center[0],
                shifted_center[1], self.half_height],
        ornObj=[0, 0, 0, 1]
      )

  def _sample_goal_in_maze(self, env, first_time=False):
    goal_pos = np.random.uniform([-15, -15], [15, 15], size=(2,))
    env._world_dict["goal_pos"] = np.concatenate([goal_pos, [0.32]])
    if first_time:
      self.goal_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_SPHERE, radius=0.8,
        rgbaColor=(1, 0, 0, 1)
      )
      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=self.goal_visual_id,
        basePosition=env._world_dict["goal_pos"]
      )
    else:
      env.pybullet_client.resetBasePositionAndOrientation(
        self.goal_visual_id, env._world_dict["goal_pos"], ornObj=[0, 0, 0, 1])

  def _generate_convex_blocks_sparse(self, env):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """
    if self._created:
      self._move_block_pos(env)
      return
    self.box_ids = []
    self.half_height = 0.7
    self.half_length = 0.3 / (2 * math.sqrt(2))
    if self.random_shape:
      delta_half_length = np.random.uniform(
        low=-0.01, high=0.2, size=(50, 2))
      delta_half_height = np.random.uniform(low=-0.25, high=0.25, size=50)

    block_centers = np.split(np.random.uniform(
      [2.0, -3.0], [30, 3.0], size=(50, 2)), 50)

    for i, center in enumerate(block_centers):
      # We want the blocks to be in front of the robot.
      shifted_center = center.reshape(2) + np.array([2.5, -3.0])
      if self.random_shape:
        half_height = self.half_height + delta_half_height[i]
        half_length = self.half_length + delta_half_length[i]
        box_id = env.pybullet_client.createCollisionShape(
          env.pybullet_client.GEOM_BOX, halfExtents=[half_length[0] * 1.7 + 0.05, half_length[1] * 1.7 + 0.05, half_height])

        box_visual_id = env.pybullet_client.createVisualShape(
          env.pybullet_client.GEOM_BOX, halfExtents=[
            half_length[0] * 1.7, half_length[1] * 1.7, half_height],
          rgbaColor=(0.1, 0.1, 0.1, 1))
        b_id = env.pybullet_client.createMultiBody(
          baseMass=0,
          baseCollisionShapeIndex=box_id,
          baseVisualShapeIndex=box_visual_id,
          basePosition=[shifted_center[0], shifted_center[1], half_height * 0.5])
      else:
        box_id = env.pybullet_client.createCollisionShape(
          env.pybullet_client.GEOM_BOX, halfExtents=[self.half_length * 1.7 + 0.05, self.half_length * 1.7 + 0.05, self.half_height])

        box_visual_id = env.pybullet_client.createVisualShape(
          env.pybullet_client.GEOM_BOX, halfExtents=[
            self.half_length * 1.7, self.half_length * 1.7, self.half_height],
          rgbaColor=(0.1, 0.1, 0.1, 1))
        b_id = env.pybullet_client.createMultiBody(
          baseMass=0,
          baseCollisionShapeIndex=box_id,
          baseVisualShapeIndex=box_visual_id,
          basePosition=[shifted_center[0], shifted_center[1], self.half_height])
      self.box_ids.append([shifted_center, b_id])
      # print(env.pybullet_client.getBasePositionAndOrientation(b_id))

    # Fench
    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30 + 0.05, self.half_length + 0.05, self.half_height * 3
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, self.half_length, self.half_height * 3],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, 3.1, self.half_height * 3]
    )

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30 + 0.05, self.half_length + 0.05, self.half_height * 3
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, self.half_length, self.half_height * 3],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, -3.1, self.half_height * 3]
    )

    self._created = True

  def _generate_stairs(self, env):
    sth = 0.10
    boxHalfLength = 2
    boxHalfWidth = 25
    boxHalfHeight = 0.2
    sh_colBox = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])
    stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                basePosition=[2.75, 0, -0.2+1*sth], baseOrientation=[0.0, 0.0, 0.0, 1])
    env.pybullet_client.changeVisualShape(
      stair, -1, rgbaColor=[1, 0, 0, 1])
    stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                basePosition=[2.75+0.44, 0, -0.2+2*sth], baseOrientation=[0.0, 0.0, 0.0, 1])
    env.pybullet_client.changeVisualShape(
      stair, -1, rgbaColor=[0, 1, 0, 1])
    stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                basePosition=[2.75+0.88, 0, -0.2+3*sth], baseOrientation=[0.0, 0.0, 0.0, 1])
    env.pybullet_client.changeVisualShape(
      stair, -1, rgbaColor=[0, 0, 1, 1])
    stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                basePosition=[2.75+1.32, 0, -0.2+4*sth], baseOrientation=[0.0, 0.0, 0.0, 1])
    env.pybullet_client.changeVisualShape(
      stair, -1, rgbaColor=[1, 1, 1, 1])
    stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                basePosition=[2.75+0.44*4, 0, -0.2+3*sth], baseOrientation=[0.0, 0.0, 0.0, 1])
    env.pybullet_client.changeVisualShape(
      stair, -1, rgbaColor=[0, 0, 1, 1])
    stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                basePosition=[2.75+0.44*5, 0, -0.2+2*sth], baseOrientation=[0.0, 0.0, 0.0, 1])
    env.pybullet_client.changeVisualShape(
      stair, -1, rgbaColor=[0, 1, 0, 1])
    stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                basePosition=[2.75+0.44*6, 0, -0.2+1*sth], baseOrientation=[0.0, 0.0, 0.0, 1])
    env.pybullet_client.changeVisualShape(
      stair, -1, rgbaColor=[1, 0, 0, 1])
    if self.goal:
      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_SPHERE, radius=0.5,
        rgbaColor=(1, 0, 0, 1)
      )
      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=box_visual_id,
        basePosition=GOAL_POS["stairs"]
      )
      env._world_dict["goal_pos"] = GOAL_POS["stairs"]
    self._created = True

  def _generate_multi_stairs(self, env):
    num_stairs = np.random.randint(low=1, high=6)
    sth = 0.05
    boxHalfLength = 2
    boxHalfWidth = 25
    boxHalfHeight = 0.2
    for i in range(num_stairs):
      noise = 8 * np.random.rand() - 4 if i > 0 else 0
      h_noise = np.random.rand() * 0.02 - 0.01
      sh_colBox = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])
      stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                  basePosition=[noise + 6.75 * i + 2.75, 0, -0.2+1*(sth + h_noise)], baseOrientation=[0.0, 0.0, 0.0, 1])
      env.pybullet_client.changeVisualShape(
        stair, -1, rgbaColor=[1, 0, 0, 1])
      stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                  basePosition=[noise + 6.75 * i + 2.75+0.44, 0, -0.2+2*(sth + h_noise)], baseOrientation=[0.0, 0.0, 0.0, 1])
      env.pybullet_client.changeVisualShape(
        stair, -1, rgbaColor=[0, 1, 0, 1])
      stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                  basePosition=[noise + 6.75 * i + 2.75+0.88, 0, -0.2+3*(sth + h_noise)], baseOrientation=[0.0, 0.0, 0.0, 1])
      env.pybullet_client.changeVisualShape(
        stair, -1, rgbaColor=[0, 0, 1, 1])
      stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                  basePosition=[noise + 6.75 * i + 2.75+1.32, 0, -0.2+4*(sth + h_noise)], baseOrientation=[0.0, 0.0, 0.0, 1])
      env.pybullet_client.changeVisualShape(
        stair, -1, rgbaColor=[1, 1, 1, 1])
      stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                  basePosition=[noise + 6.75 * i + 2.75+0.44*4, 0, -0.2+3*(sth + h_noise)], baseOrientation=[0.0, 0.0, 0.0, 1])
      env.pybullet_client.changeVisualShape(
        stair, -1, rgbaColor=[0, 0, 1, 1])
      stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                  basePosition=[noise + 6.75 * i + 2.75+0.44*5, 0, -0.2+2*(sth + h_noise)], baseOrientation=[0.0, 0.0, 0.0, 1])
      env.pybullet_client.changeVisualShape(
        stair, -1, rgbaColor=[0, 1, 0, 1])
      stair = env.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                  basePosition=[noise + 6.75 * i + 2.75+0.44*6, 0, -0.2+1*(sth + h_noise)], baseOrientation=[0.0, 0.0, 0.0, 1])
      env.pybullet_client.changeVisualShape(
        stair, -1, rgbaColor=[1, 0, 0, 1])

    if self.goal:
      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_SPHERE, radius=0.5,
        rgbaColor=(1, 0, 0, 1)
      )
      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=box_visual_id,
        basePosition=GOAL_POS["multi_stairs"]
      )
      env._world_dict["goal_pos"] = GOAL_POS["multi_stairs"]
    self._created = True

  def _generate_field(self, env, num_rows=None, num_columns=None):
    # print("Generate Field")
    if num_rows == None:
      num_rows = numHeightfieldRows
    if num_columns == None:
      num_columns = numHeightfieldRows

    if not self.terrain_created:
      self.heightfieldData = [0] * num_rows * num_columns

    heightPerturbationRange = self.height_range

    heightPerturbationRange = heightPerturbationRange
    # if heightfieldSource == useProgrammatic:

    for j in range(int(num_columns / 2)):
      for i in range(int(num_rows / 2)):
        height = random.uniform(0, heightPerturbationRange)
        self.heightfieldData[2 * i +
                             2 * j * num_rows] = height
        self.heightfieldData[2 * i + 1 +
                             2 * j * num_rows] = height
        self.heightfieldData[2 * i + (2 * j + 1) *
                             num_rows] = height
        self.heightfieldData[2 * i + 1 + (2 * j + 1) *
                             num_rows] = height
    for j in range(-5, 5):
      for i in range(-5, 5):
        x = int(num_rows / 4) + i
        y = int(num_columns / 4) + j
        # print(x, y)
        self.heightfieldData[2 * x +
                             2 * y * num_rows] = 0
        self.heightfieldData[2 * x + 1 +
                             2 * y * num_rows] = 0
        self.heightfieldData[2 * x + (2 * y + 1) *
                             num_rows] = 0
        self.heightfieldData[2 * x + 1 + (2 * y + 1) *
                             num_rows] = 0

    if self.terrain_created:
      return
    else:
      self.terrainShape = env.pybullet_client.createCollisionShape(
        shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
        meshScale=[.12, .12, 1.0],
        heightfieldTextureScaling=0,
        heightfieldData=self.heightfieldData,
        numHeightfieldRows=num_rows,
        numHeightfieldColumns=num_columns)
    terrain = env.pybullet_client.createMultiBody(0, self.terrainShape)
    env.pybullet_client.resetBasePositionAndOrientation(
      terrain, [0.0, 0.0, 0.0], [0, 0, 0, 1])

    env.pybullet_client.changeVisualShape(terrain,
                                          -1,
                                          rgbaColor=[0.1, 0.1, 0.1, 1],
                                          specularColor=[0.1, 0.1, 0.1, 1]
                                          )
    env._world_dict["terrain"] = terrain
    # env.pybullet_client.configureDebugVisualizer(
    #     env.pybullet_client.COV_ENABLE_RENDERING, 1)
    self.terrain_created = True

  def _move_block_pos_easy(self, env):
    for b_id, pos in self.box_ids:
      x, y, z = pos
      env.pybullet_client.resetBasePositionAndOrientation(
        b_id,
        posObj=[
          x + 0.5 * (np.random.rand() - 0.5) * self.prob,
          y + 2 * (np.random.rand() - 0.5) * self.prob,
          z],
        ornObj=[0, 0, 0, 1]
      )

  def _generate_convex_blocks_sparse_easy(self, env):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """
    if self._created:
      # if self.multiple:
      self._move_block_pos_easy(env)
      return

    self.box_ids = []
    half_length = 0.25
    self.half_height = 1.0

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        half_length+0.05, 0.4 + 0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[half_length, 0.4, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    b_id = env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[3, 0.75, self.half_height*0.5]
    )
    self.box_ids.append((b_id, (3, 0.75, self.half_height*0.5)))

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        half_length+0.05, 0.4 + 0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[half_length, 0.4, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    b_id = env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[3, -0.75, self.half_height*0.5]
    )
    self.box_ids.append((b_id, (3, -0.75, self.half_height*0.5)))

    for i in range(7):
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[half_length+0.05, 0.8+0.05, self.half_height * 0.5])

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[
          half_length, 0.8, self.half_height * 0.5],
        rgbaColor=(0.1, 0.1, 0.1, 1))

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[6 + i * 7, 0, self.half_height * 0.5])
      self.box_ids.append((b_id, (6 + i * 7, 0, 0.5 * self.half_height)))

      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          half_length+0.05, 0.8 + 0.05, self.half_height*0.5
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[half_length, 0.8, self.half_height*0.5],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[9 + i * 7, -1.3, self.half_height*0.5]
      )
      self.box_ids.append(
        (b_id, (9 + i * 7, -1.3, self.half_height*0.5)))

      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          half_length+0.05, 0.8 + 0.05, self.half_height*0.5
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[half_length, 0.8, self.half_height*0.5],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[9 + i * 7, 1.3, self.half_height*0.5]
      )
      self.box_ids.append((b_id, (9 + i * 7, 1.3, self.half_height*0.5)))

    # Fench
    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30+0.05, half_length+0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height * 0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, 2.3, self.half_height * 0.5]
    )

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30 + 0.05, half_length + 0.05, self.half_height * 0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height * 0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, -2.3, self.half_height * 0.5]
    )
    self._created = True

  def _generate_terrain(self, env, height_perturbation_range=0.05):
    height_perturbation_range = height_perturbation_range
    if self._terrain_type == TerrainType.RANDOM_HILL:
      terrain_shape = env.pybullet_client.createCollisionShape(
        shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
        meshScale=[.2, .2, .2],
        flags=1,
        fileName="heightmaps/ground0.txt",
        heightfieldTextureScaling=128)
      terrain = env.pybullet_client.createMultiBody(0, terrain_shape)
      textureId = env.pybullet_client.loadTexture("grass.png")
      env.pybullet_client.changeVisualShape(
        terrain, -1, textureUniqueId=textureId, flags=1)
      env.pybullet_client.resetBasePositionAndOrientation(
        terrain, [1, 0, 2], [0, 0, 0, 1])

    elif self._terrain_type == TerrainType.RANDOM_MOUNT or self._terrain_type == TerrainType.GOAL_MOUNT:
      terrain_shape = env.pybullet_client.createCollisionShape(
        shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
        meshScale=[0.1, 0.1, 15 * MOUNT_LEVEL[self.mount_level]],
        flags=1,
        fileName=FLAG_TO_FILENAME["mounts"])
      terrain = env.pybullet_client.createMultiBody(0, terrain_shape)
      textureId = env.pybullet_client.loadTexture(
        "heightmaps/gimp_overlay_out.png")
      env.pybullet_client.changeVisualShape(
        terrain, -1, textureUniqueId=textureId, flags=1)
      # Move Origin A little bit to start at Flat Area
      env.pybullet_client.resetBasePositionAndOrientation(
        terrain, [2, 2, 2 * MOUNT_LEVEL[self.mount_level]], [0, 0, 0, 1])
      if self._terrain_type == TerrainType.GOAL_MOUNT:
        box_visual_id = env.pybullet_client.createVisualShape(
          env.pybullet_client.GEOM_SPHERE, radius=0.8 *
          MOUNT_LEVEL[self.mount_level],
          rgbaColor=(1, 0, 0, 1)
        )
        visual_pos = GOAL_POS["mounts"][self.mount_level]
        env.world_dict["goal_pos"] = visual_pos
        # visual_pos[1] -= 0.5
        env.pybullet_client.createMultiBody(
          baseMass=0,
          baseVisualShapeIndex=box_visual_id,
          basePosition=visual_pos
        )

    elif self._terrain_type == TerrainType.MAZE:
      raise NotImplementedError("Maze Terrain not implemented")
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
      self._sample_goal_in_maze(env, first_time=True)
    else:
      raise NotImplementedError

    self.terrain_shape = terrain_shape
    env.pybullet_client.changeVisualShape(
      terrain, -1, rgbaColor=[1, 1, 1, 1])
    if self._terrain_type is not TerrainType.MAZE:
      env.pybullet_client.changeDynamics(
        terrain, -1,
        lateralFriction=env.fric_coeff[0],
        spinningFriction=env.fric_coeff[1],
        rollingFriction=env.fric_coeff[2])
    if self.subgoal:
      self._generate_mountain_subgoal(env)
    env._world_dict["terrain"] = terrain

    self.terrain_created = True

  def _move_block_and_subgoal_pos(self, env):

    self.subgoal_centers = np.random.uniform(
      [2.0, -2.2], [30, 2.2], size=(50, 2)
    )
    subgoal_centers = np.split(self.subgoal_centers, 50)

    for idx, box_id in enumerate(self.subgoal_ids):
      shifted_center = subgoal_centers[idx].reshape(2)
      env.pybullet_client.resetBasePositionAndOrientation(
        box_id,
        posObj=[shifted_center[0], shifted_center[1], self.radius],
        ornObj=[0, 0, 0, 1]
      )

      env.pybullet_client.changeVisualShape(
        box_id,
        -1,
        rgbaColor=(1, 0.2, 0.2, 1)
      )

    block_centers = np.split(np.random.uniform(
      [2.0, -3.0], [16.0, 3.0], size=(50, 2)), 50)

    for idx, box_id in enumerate(self.box_ids):
      pos, b_id = box_id
      # x, y = pos
      shifted_center = block_centers[idx].reshape(2)
      env.pybullet_client.resetBasePositionAndOrientation(
        b_id,
        posObj=[shifted_center[0],
                shifted_center[1], self.half_height],
        ornObj=[0, 0, 0, 1]
      )

  def _generate_convex_blocks_sparse_hard_with_subgoal(self, env):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """
    if self._created:
      self._move_block_and_subgoal_pos(env)
      return
    self.box_ids = []
    half_length = 0.3 / (2 * math.sqrt(2))
    self.half_height = 2

    self.half_height = 0.7

    block_centers = np.split(np.random.uniform(
      [2.0, -3.0], [16.0, 3.0], size=(50, 2)), 50)

    subgoal_centers = np.split(np.random.uniform(
      [2.0, -2.2], [30.0, 2.2], size=(50, 2)), 50)

    self.subgoal_ids = []

    for center in subgoal_centers:
      self.radius = 0.2
      shifted_center = center.reshape(2)

      subgoal_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_SPHERE,
        radius=self.radius,
        rgbaColor=(1, 0.2, 0.2, 1))

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        # baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=subgoal_visual_id,
        basePosition=[shifted_center[0], shifted_center[1], self.radius])
      self.subgoal_ids.append(b_id)

    for center in block_centers:
      # We want the blocks to be in front of the robot.
      # shifted_center = np.array(center) - [2, _GRID_WIDTH / 2]
      shifted_center = center.reshape(2)
      # Do not place blocks near the point [0, 0], where the robot will start.
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[half_length * 1.7+0.05, half_length * 1.7+0.05, self.half_height])

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[
          half_length * 1.7, half_length * 1.7, self.half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1))

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[shifted_center[0], shifted_center[1], self.half_height])
      self.box_ids.append([shifted_center, b_id])

    # Fench
    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30+0.05, half_length+0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, 3.1, self.half_height*0.5]
    )

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30+0.05, half_length+0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, -3.1, self.half_height*0.5]
    )
    self._created = True

  def _move_sphere_and_subgoal_pos(self, env):

    self.subgoal_centers = np.random.uniform(
      [2.0, -2.2], [30, 2.2], size=(50, 2)
    )
    subgoal_centers = np.split(self.subgoal_centers, 50)

    for idx, box_id in enumerate(self.subgoal_ids):
      # pos, b_id = box_id
      # x, y = pos
      shifted_center = subgoal_centers[idx].reshape(2)
      env.pybullet_client.resetBasePositionAndOrientation(
        box_id,
        posObj=[shifted_center[0], shifted_center[1], self.radius],
        ornObj=[0, 0, 0, 1]
      )

      env.pybullet_client.changeVisualShape(
        box_id,
        -1,
        rgbaColor=(1, 0.2, 0.2, 1)
      )

    self.sphere_centers = sphere_centers = np.split(np.random.uniform(
      [2.0, -3.0], [16.0, 3.0], size=(50, 2)), 50)

    for idx, box_id in enumerate(self.box_ids):
      # pos, b_id = box_id
      # x, y = pos
      shifted_center = sphere_centers[idx].reshape(2)
      env.pybullet_client.resetBasePositionAndOrientation(
        box_id,
        posObj=[shifted_center[0], shifted_center[1], self.radius],
        ornObj=[0, 0, 0, 1]
      )

  def _generate_spheres_and_subgoal(self, env):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """
    if self._created:
      self._move_sphere_and_subgoal_pos(env)
      # if self.multiple:
      #     self._move_block_pos(env)
      return

    self.half_height = 0.7

    subgoal_centers = np.split(np.random.uniform(
      [2.0, -2.2], [30.0, 2.2], size=(50, 2)), 50)

    self.subgoal_ids = []

    for center in subgoal_centers:
      self.radius = 0.2
      shifted_center = center.reshape(2)

      # No Collision
      # subgoal_id = env.pybullet_client.createCollisionShape(
      #     env.pybullet_client.GEOM_BOX, halfExtents=[half_length * 1.7+0.05, half_length * 1.7+0.05, self.half_height])

      subgoal_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_SPHERE,
        radius=self.radius,
        rgbaColor=(1, 0.2, 0.2, 1))

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=subgoal_visual_id,
        basePosition=[shifted_center[0], shifted_center[1], self.radius])
      self.subgoal_ids.append(b_id)

    self.sphere_ids = []

    sphere_centers = np.split(np.random.uniform(
      [2.0, -3.0], [16.0, 3.0], size=(50, 2)), 50)

    for center in sphere_centers:
      # We want the blocks to be in front of the robot.
      # shifted_center = np.array(center) - [2, _GRID_WIDTH / 2]
      shifted_center = center.reshape(2)

      sphere_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_SPHERE,
        radius=self.radius,
      )

      sphere_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_SPHERE,
        radius=self.radius,
        rgbaColor=(0.2, 0.2, 0.1, 1))

      s_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=sphere_id,
        baseVisualShapeIndex=sphere_visual_id,
        basePosition=[shifted_center[0], shifted_center[1], self.radius])

      self.box_ids.append(s_id)
      # print(env.pybullet_client.getBasePositionAndOrientation(b_id))

    half_length = 0.3 / (2 * math.sqrt(2))
    # Fench
    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30+0.05, half_length+0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, 3.1, self.half_height*0.5]
    )

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30+0.05, half_length+0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, -3.1, self.half_height*0.5]
    )
    self._created = True
    # print("create_time: ", time.time() - create_time)

  def _generate_mountain_subgoal(self, env):
    for pos in SUBGOAL_POS:
      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_SPHERE, radius=0.5,
        rgbaColor=(1, 0, 0, 1)
      )
      env.pybullet_client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=box_visual_id,
        basePosition=pos
      )
    env._world_dict["subgoals"] = SUBGOAL_POS
    env._world_dict["subgoals_achieved"] = [False, False]

  def _move_block_and_subgoal_pos_easy(self, env):
    for b_id, pos in self.box_ids:
      x, y = pos
      env.pybullet_client.resetBasePositionAndOrientation(
        b_id,
        posObj=[
          x + 0.5 * (np.random.rand() - 0.5) * self.prob,
          y + 2 * (np.random.rand() - 0.5) * self.prob,
          self.half_height * 0.5],
        ornObj=[0, 0, 0, 1]
      )

    self.subgoal_centers = np.random.uniform(
      [2.0, -2.2], [30, 2.2], size=(50, 2)
    )
    subgoal_centers = np.split(self.subgoal_centers, 50)

    for idx, box_id in enumerate(self.subgoal_ids):
      # pos, b_id = box_id
      # x, y = pos
      shifted_center = subgoal_centers[idx].reshape(2)
      env.pybullet_client.resetBasePositionAndOrientation(
        box_id,
        posObj=[shifted_center[0], shifted_center[1], self.radius],
        ornObj=[0, 0, 0, 1]
      )

      env.pybullet_client.changeVisualShape(
        box_id,
        -1,
        rgbaColor=(1, 0.2, 0.2, 1)
      )

  def _generate_convex_blocks_sparse_easy_with_subgoal(self, env):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """
    if self._created:
      # if self.multiple:
      self._move_block_and_subgoal_pos_easy(env)
      return

    subgoal_centers = np.split(np.random.uniform(
      [2.0, -2.2], [30.0, 2.2], size=(50, 2)), 50)

    self.subgoal_ids = []

    for center in subgoal_centers:
      self.radius = 0.2
      shifted_center = center.reshape(2)

      # No Collision
      # subgoal_id = env.pybullet_client.createCollisionShape(
      #     env.pybullet_client.GEOM_BOX, halfExtents=[half_length * 1.7+0.05, half_length * 1.7+0.05, self.half_height])

      subgoal_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_SPHERE,
        radius=self.radius,
        rgbaColor=(1, 0.2, 0.2, 1))

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=subgoal_visual_id,
        basePosition=[shifted_center[0], shifted_center[1], self.radius])
      self.subgoal_ids.append(b_id)

    self.box_ids = []
    # half_length = 0.3 / (2 * math.sqrt(2))
    # self.half_height = 2
    half_length = 0.25

    # self.half_height = 2
    # if self.multiple:
    self.half_height = 1.0

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        half_length+0.05, 0.4 + 0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[half_length, 0.4, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    b_id = env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[3, 0.75, self.half_height*0.5]
    )
    self.box_ids.append((b_id, (3, 0.75)))

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        half_length+0.05, 0.4 + 0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[half_length, 0.4, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    b_id = env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[3, -0.75, self.half_height*0.5]
    )
    self.box_ids.append((b_id, (3, -0.75)))

    for i in range(7):
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[half_length+0.05, 0.8+0.05, self.half_height * 0.5])

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[
          half_length, 0.8, self.half_height * 0.5],
        rgbaColor=(0.1, 0.1, 0.1, 1))

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[6 + i * 7, 0, self.half_height * 0.5])
      self.box_ids.append((b_id, (6 + i * 7, 0)))

      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          half_length+0.05, 0.8 + 0.05, self.half_height*0.5
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[half_length, 0.8, self.half_height*0.5],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[9 + i * 7, -1.3, self.half_height*0.5]
      )
      self.box_ids.append((b_id, (9 + i * 7, -1.3)))

      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          half_length+0.05, 0.8 + 0.05, self.half_height*0.5
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[half_length, 0.8, self.half_height*0.5],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[9 + i * 7, 1.3, self.half_height*0.5]
      )
      self.box_ids.append((b_id, (9 + i * 7, 1.3)))

    # Fench
    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30+0.05, half_length+0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, 2.3, self.half_height*0.5]
    )

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30+0.05, half_length+0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, -2.3, self.half_height*0.5]
    )
    self._created = True
    # print("create_time: ", time.time() - create_time)

  def _move_block_thin_wide_and_subgoal_pos(self, env, with_subgoal=False):
    for b_id, pos in self.wide_box_ids:
      x, y = pos
      env.pybullet_client.resetBasePositionAndOrientation(
        b_id,
        posObj=[
          x + np.random.rand() * 2 * self.prob,
          y + np.random.rand() * 0 * self.prob,
          self.half_height * 0.5],
        ornObj=[0, 0, 0, 1]
      )

    block_centers = np.split(np.random.uniform(
      [2.0, -3.0], [16.0, 3.0], size=(30, 2)), 30)

    for idx, box_id in enumerate(self.thin_box_ids):
      pos, b_id = box_id
      # x, y = pos
      shifted_center = block_centers[idx].reshape(2)
      env.pybullet_client.resetBasePositionAndOrientation(
        b_id,
        posObj=[shifted_center[0],
                shifted_center[1], self.half_height],
        ornObj=[0, 0, 0, 1]
      )

    if with_subgoal:
      self.subgoal_centers = np.random.uniform(
        [2.0, -2.2], [30, 2.2], size=(50, 2)
      )
      subgoal_centers = np.split(self.subgoal_centers, 50)

      for idx, box_id in enumerate(self.subgoal_ids):
        # pos, b_id = box_id
        # x, y = pos
        shifted_center = subgoal_centers[idx].reshape(2)
        env.pybullet_client.resetBasePositionAndOrientation(
          box_id,
          posObj=[shifted_center[0], shifted_center[1], self.radius],
          ornObj=[0, 0, 0, 1]
        )

        env.pybullet_client.changeVisualShape(
          box_id,
          -1,
          rgbaColor=(1, 0.2, 0.2, 1)
        )

  def _generate_convex_blocks_thin_wide(self, env, with_subgoal=False):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """
    if self._created:
      # if self.multiple:
      self._move_block_thin_wide_and_subgoal_pos(env, with_subgoal)
      return

    if with_subgoal:
      subgoal_centers = np.split(np.random.uniform(
        [2.0, -2.2], [30.0, 2.2], size=(50, 2)), 50)

      self.subgoal_ids = []
      for center in subgoal_centers:
        self.radius = 0.2
        shifted_center = center.reshape(2)

        subgoal_visual_id = env.pybullet_client.createVisualShape(
          env.pybullet_client.GEOM_SPHERE,
          radius=self.radius,
          rgbaColor=(1, 0.2, 0.2, 1))

        b_id = env.pybullet_client.createMultiBody(
          baseMass=0,
          baseVisualShapeIndex=subgoal_visual_id,
          basePosition=[shifted_center[0], shifted_center[1], self.radius])
        self.subgoal_ids.append(b_id)

    self.box_ids = []
    self.wide_box_ids = []
    half_length = 0.25

    self.half_height = 1.0

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        half_length+0.05, 0.4 + 0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[half_length, 0.4, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    b_id = env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[2, 0.75, self.half_height*0.5]
    )
    self.wide_box_ids.append((b_id, (2, 0.75)))

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        half_length+0.05, 0.4 + 0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[half_length, 0.4, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    b_id = env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[2, -0.75, self.half_height*0.5]
    )
    self.wide_box_ids.append((b_id, (2, -0.75)))

    for i in range(7):
      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[half_length+0.05, 0.8 + 0.05, self.half_height * 0.5])

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[
          half_length, 0.8, self.half_height * 0.5],
        rgbaColor=(0.1, 0.1, 0.1, 1))

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[5 + i * 7, 0, self.half_height * 0.5])
      self.wide_box_ids.append((b_id, (5 + i * 7, 0)))

      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          half_length+0.05, 0.8 + 0.05, self.half_height * 0.5
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[half_length, 0.8, self.half_height * 0.5],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[8 + i * 7, -1.8, self.half_height * 0.5]
      )
      self.wide_box_ids.append((b_id, (8 + i * 7, -1.8)))

      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[
          half_length+0.05, 0.8 + 0.05, self.half_height * 0.5
        ]
      )

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX,
        halfExtents=[half_length, 0.8, self.half_height * 0.5],
        rgbaColor=(0.1, 0.1, 0.1, 1)
      )

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[8 + i * 7, 1.8, self.half_height * 0.5]
      )
      self.wide_box_ids.append((b_id, (8 + i * 7, 1.8)))

    self.thin_box_ids = []
    half_length = 0.3 / (2 * math.sqrt(2))

    # self.half_height = 0.7

    block_centers = np.split(np.random.uniform(
      [2.0, -2.0], [30.0, 2.0], size=(30, 2)), 30)

    for center in block_centers:
      # We want the blocks to be in front of the robot.
      # shifted_center = np.array(center) - [2, _GRID_WIDTH / 2]
      shifted_center = center.reshape(2)

      box_id = env.pybullet_client.createCollisionShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[half_length * 1.7 + 0.05, half_length * 1.7 + 0.05, self.half_height])

      box_visual_id = env.pybullet_client.createVisualShape(
        env.pybullet_client.GEOM_BOX, halfExtents=[
          half_length * 1.7, half_length * 1.7, self.half_height],
        rgbaColor=(0.1, 0.1, 0.1, 1))

      b_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=box_id,
        baseVisualShapeIndex=box_visual_id,
        basePosition=[shifted_center[0], shifted_center[1], self.half_height])
      self.thin_box_ids.append([shifted_center, b_id])
      self.box_ids.append([shifted_center, b_id])

    # Fench
    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30+0.05, half_length + 0.05, self.half_height * 3
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height * 3],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, 3.0, self.half_height * 3]
    )

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30 + 0.05, half_length + 0.05, self.half_height * 3
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height * 3],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, -3.0, self.half_height * 3]
    )
    self._created = True
    # print("create_time: ", time.time() - create_time)

  def _move_chair_desk_and_subgoal_pos(self, env, with_subgoal=False):

    self.poisson_disc = PoissonDisc2D(26, 6, 1.1, 50)
    generated_centers = self.poisson_disc.generate()
    # print(len(generated_centers))

    # generated_idx = np.arange(45)
    np.random.shuffle(generated_centers)
    # print(generated_centers)
    # print(len(generated_centers))
    # print(self.poisson_disc._max_sample_size)

    # chair_centers = np.split(np.random.uniform(
    #     [2.0, -3.0], [16.0, 3.0], size=(30, 2)), 30)

    for idx, chair_id in enumerate(self.chair_ids):
      # pos, b_id = box_id
      # x, y = pos
      # shifted_center = chair_centers[idx].reshape(2)
      shifted_center = generated_centers[idx] + np.array([2.5, -3.0])
      env.pybullet_client.resetBasePositionAndOrientation(
        chair_id,
        posObj=[shifted_center[0],
                shifted_center[1], 0.34],
        ornObj=[1, 0, 0, 1]
      )

    # desk_centers = np.split(np.random.uniform(
    #     [2.0, -2.0], [30.0, 2.0], size=(15, 2)), 15)

    for idx, desk_id in enumerate(self.desk_ids):
      # pos, b_id = box_id
      # x, y = pos
      # shifted_center = desk_centers[idx].reshape(2)
      shifted_center = generated_centers[idx +
                                         50] + np.array([2.5, -3.0])
      env.pybullet_client.resetBasePositionAndOrientation(
        desk_id,
        posObj=[shifted_center[0],
                shifted_center[1], 0.24],
        ornObj=[1, 0, 0, 1]
      )

    if with_subgoal:
      self.subgoal_centers = np.random.uniform(
        [2.0, -2.2], [30, 2.2], size=(50, 2)
      )
      subgoal_centers = np.split(self.subgoal_centers, 50)

      for idx, box_id in enumerate(self.subgoal_ids):
        # pos, b_id = box_id
        # x, y = pos
        shifted_center = subgoal_centers[idx].reshape(2)
        env.pybullet_client.resetBasePositionAndOrientation(
          box_id,
          posObj=[shifted_center[0], shifted_center[1], self.radius],
          ornObj=[0, 0, 0, 1]
        )

        env.pybullet_client.changeVisualShape(
          box_id,
          -1,
          rgbaColor=(1, 0.2, 0.2, 1)
        )

  def _generate_chair_desk(self, env, with_subgoal=False):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """
    if self._created:
      # if self.multiple:
      self._move_chair_desk_and_subgoal_pos(env, with_subgoal)
      return

    self.poisson_disc = PoissonDisc2D(24, 6, 1.2, 50)

    if with_subgoal:
      subgoal_centers = np.split(np.random.uniform(
        [2.0, -2.2], [30.0, 2.2], size=(50, 2)), 50)

      self.subgoal_ids = []
      for center in subgoal_centers:
        self.radius = 0.2
        shifted_center = center.reshape(2)

        subgoal_visual_id = env.pybullet_client.createVisualShape(
          env.pybullet_client.GEOM_SPHERE,
          radius=self.radius,
          rgbaColor=(1, 0.2, 0.2, 1))

        b_id = env.pybullet_client.createMultiBody(
          baseMass=0,
          baseVisualShapeIndex=subgoal_visual_id,
          basePosition=[shifted_center[0], shifted_center[1], self.radius])
        self.subgoal_ids.append(b_id)

    self.wide_box_ids = []
    half_length = 0.25

    self.half_height = 1.0

    self.chair_ids = []
    half_length = 0.3 / (2 * math.sqrt(2))
    chair_centers = np.split(np.random.uniform(
      [2.0, -2.0], [30.0, 2.0], size=(50, 2)), 50)

    for center in chair_centers:
      # We want the blocks to be in front of the robot.
      # shifted_center = np.array(center) - [2, _GRID_WIDTH / 2]
      shifted_center = center.reshape(2)

      chair_id = env.pybullet_client.loadURDF(
        "chair/chair.urdf", [0, 0, 1], globalScaling=8, useFixedBase=True)
      env.pybullet_client.resetBasePositionAndOrientation(chair_id, [shifted_center[0],
                                                                     shifted_center[1], 0.34], ornObj=[1, 0, 0, 1])
      self.chair_ids.append(chair_id)

    self.desk_ids = []
    half_length = 0.3 / (2 * math.sqrt(2))
    desk_centers = np.split(np.random.uniform(
      [2.0, -2.0], [30.0, 2.0], size=(30, 2)), 30)

    for center in desk_centers:
      # We want the blocks to be in front of the robot.
      # shifted_center = np.array(center) - [2, _GRID_WIDTH / 2]
      shifted_center = center.reshape(2)

      desk_id = env.pybullet_client.loadURDF(
        "desk/desk.urdf", [0, 0, 1], globalScaling=17, useFixedBase=True)
      env.pybullet_client.resetBasePositionAndOrientation(
        desk_id, [shifted_center[0],
                  shifted_center[1], 0.34], ornObj=[1, 0, 0, 1])
      self.desk_ids.append(desk_id)

    # Fench
    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30+0.05, half_length+0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, 3.0, self.half_height*0.5]
    )

    box_id = env.pybullet_client.createCollisionShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[
        30+0.05, half_length+0.05, self.half_height*0.5
      ]
    )

    box_visual_id = env.pybullet_client.createVisualShape(
      env.pybullet_client.GEOM_BOX,
      halfExtents=[30, half_length, self.half_height*0.5],
      rgbaColor=(0.1, 0.1, 0.1, 1)
    )

    env.pybullet_client.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=box_id,
      baseVisualShapeIndex=box_visual_id,
      basePosition=[15, -3.0, self.half_height*0.5]
    )
    # self.terrain_created = True
    # print("create_time: ", time.time() - create_time)
