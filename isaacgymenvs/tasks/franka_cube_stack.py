# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
# import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym.torch_utils import *

from typing import Dict, Any, Tuple
# import sapien.core as sapien
# from sapien.core import SceneConfig, Pose

# A force sensor, measures force and torque. For each force sensor,
# its measurements are represented by a tensor with 6 elements of dtype float32.
# The first 3 element are the force measurements and the last 3 are the torque measurements.
N_ELEM_PER_FORCE_SENSOR = 6

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaCubeStack(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.z_size = 32 
        self.cfg["env"]["numObservations"] = 14 + self.z_size
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cube1_state = None           # Initial state of cube1 for the current env
        self._init_cube2_state = None           # Initial state of cube1 for the current env
        self._init_cube3_state = None           # Initial state of cube1 for the current env
        self._init_cube4_state = None           # Initial state of cube1 for the current env
        self._init_cube5_state = None           # Initial state of cube1 for the current env
        self._init_cube6_state = None           # Initial state of cube1 for the current env
        self._init_cube7_state = None           # Initial state of cube1 for the current env
        self._init_cube8_state = None           # Initial state of cube1 for the current env
        self._init_cube9_state = None           # Initial state of cube1 for the current env

        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cube1_state = None                # Current state of cube1 for the current env
        self._cube2_state = None                # Current state of cube1 for the current env
        self._cube3_state = None                # Current state of cube1 for the current env
        self._cube4_state = None                # Current state of cube1 for the current env
        self._cube5_state = None                # Current state of cube1 for the current env
        self._cube6_state = None                # Current state of cube1 for the current env
        self._cube7_state = None                # Current state of cube1 for the current env
        self._cube8_state = None                # Current state of cube1 for the current env
        self._cube9_state = None                # Current state of cube1 for the current env

        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cube1_id = None                   # Actor ID corresponding to cube1 for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)


        self.actions = torch.zeros((self.num_envs, self.num_actions)).to(self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions)).to(self.device)
        self.first_step_torques = torch.zeros((self.num_envs, self.num_actions)).to(self.device)
        self.mean_gripper_torque = torch.zeros((self.num_envs, 1)).to(self.device)
        self.cartesian_targets = torch.ones((self.num_envs, 3)).to(self.device)
        self.delta_actions = torch.zeros((self.num_envs, self.num_actions)).to(self.device)


        self.franka_default_dof_pos = to_torch(
            [-0.8, 1.28, 0, -1.6, 0, 0, 0., 0.87, 0.87, 0.87, 0.87], device=self.device
        )

        # Set control limits
        self.cmd_limit = self._franka_effort_limits[:].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.my_count = 0
        # Refresh tensors
        self._refresh()



    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def transform_to_base_frame(self, target_ee_frame_pos,
                                target_ee_frame_quat,
                                prev_base_frame_ee_pos,
                                prev_base_frame_ee_quat):
        target_base_frame_pos = quat_rotate(prev_base_frame_ee_quat, target_ee_frame_pos) + prev_base_frame_ee_pos
        target_base_frame_quat = quat_mul(target_ee_frame_quat, prev_base_frame_ee_quat)
        return target_base_frame_pos, target_base_frame_quat


    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/happybot_arms_v1_0/urdf/happy_arms_gym.urdf"

        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False

        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        # asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = 3 #effort mode
        asset_options.use_mesh_materials = True
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params.resolution = 300000
        # asset_options.vhacd_params.max_convex_hulls = 50
        # asset_options.vhacd_params.max_num_vertices_per_ch = 64

        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        sensor_pose = gymapi.Transform()
        sensor_pose.p = gymapi.Vec3(0.08, 0.0, 0.)
        self.gym.create_asset_force_sensor(
            franka_asset, 9, sensor_pose
        )

        stiffness = to_torch([0] * 11, dtype=torch.float, device=self.device)
        damping = to_torch([1.617, 0.893, 1.553, 0.64, 0.93, 1.055, 0.07] + [0.001] * 4, dtype=torch.float, device=self.device)
        armature = to_torch([0.078, 0.862, 0.395, 0.303, 0.076, 0.074, 0.003] + [0.001] * 4, dtype=torch.float, device=self.device)
        friction = to_torch([0., 0.003, 0., 0.056, 0.083, 0.018, 0.017] + [0.001] * 4, dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [-0., 0.0, 1.]
        table_thickness = 0.0001
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.7, 0.7, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.01, 0.01, table_stand_height], table_opts)

        self.cubeA_size = 0.04
        self.cube1_size = 0.15

        # cube1_asset = self.gym.load_asset(self.sim, '', "/home/raphael/media/bensadoun/IsaacGymEnvs_v4_multicube/assets/urdf/objects/tray/urdf/tray.urdf", bottle_opts)
        cube1_color = gymapi.Vec3(1., 1., 1.)

        #
        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(11):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = stiffness[i]
                franka_dof_props['damping'][i] = damping[i]
                franka_dof_props['friction'][i] = friction[i]
                franka_dof_props['armature'][i] = armature[i]


            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)

        #TODO wtf
        # self.franka_dof_speed_scales[[7, 8]] = 0.1
        # franka_dof_props['effort'][7] = 200
        # franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + 0.01)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cube1_start_pose = gymapi.Transform()
        cube1_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cube1_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        tray_stand_start_pose = gymapi.Transform()
        tray_stand_start_pose.p = gymapi.Vec3(*[-0.35, 0., 1.03])
        tray_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4 + 9    # 1 for table, table stand, cubeA, cube1
        max_agg_shapes = num_franka_shapes + 4 + 9    # 1 for table, table stand, cubeA, cube1

        self.frankas = []
        self.envs = []

        NUM_OBJECTS = 1
        self.running_epoch = 0
        self.current_episode = 0
        cubeA_assets = []
        bottle_opts = gymapi.AssetOptions()
        bottle_opts.vhacd_enabled = True
        bottle_opts.vhacd_params.resolution = 300000
        bottle_opts.vhacd_params.max_convex_hulls = 50
        bottle_opts.vhacd_params.max_num_vertices_per_ch = 64
        self.cylinder_heights = []
        self.cylinder_radius = []

        self.cube_heights = []

        self.object_heights = torch.zeros((self.num_envs, 10), device = self.device)

        fixed_colors = [(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)) for _ in range(1, 10)]
        self.object_dims = torch.zeros((self.num_envs, 10, 3), device = self.device)
        # Create environments
        for i in range(self.num_envs):

            cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

            cubeA_asset = self.gym.create_box(self.sim, *([0.05] * 3), bottle_opts)
            self.object_heights[i, 0] = 0.05
            additional_assets = []
            colors = []
            for num_cube in range(1, 10):
                depth = np.random.uniform(0.05, 0.12)
                width = np.random.uniform(0.05, 0.12)
                height = np.random.uniform(0.0, 0.4)
                additional_assets.append(self.gym.create_box(self.sim, depth, width, height, bottle_opts))

                self.object_heights[i, num_cube] = height
                self.object_dims[i, num_cube] = torch.tensor([depth, width, height], device = self.device)

                color_R = fixed_colors[num_cube-1][0] #np.random.uniform(0, 1)
                color_G = fixed_colors[num_cube-1][1] #np.random.uniform(0, 1)
                color_B = fixed_colors[num_cube-1][2] #np.random.uniform(0, 1)

                colors.append(gymapi.Vec3(color_R, color_G, color_B))

            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                  1.0 + table_thickness / 2 + table_stand_height)


            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_actor)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_color = gymapi.Vec3(0.1, 0.1, 0.1)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, table_color)
            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2**1, 0)
            self._cube1_id = self.gym.create_actor(env_ptr, additional_assets[0], cube1_start_pose, "cube1", i, 2**2, 0)
            self._cube2_id = self.gym.create_actor(env_ptr, additional_assets[1], cube1_start_pose, "cube2", i, 2**3, 0)
            self._cube3_id = self.gym.create_actor(env_ptr, additional_assets[2], cube1_start_pose, "cube3", i, 2**4, 0)
            self._cube4_id = self.gym.create_actor(env_ptr, additional_assets[3], cube1_start_pose, "cube4", i, 2**5, 0)
            self._cube5_id = self.gym.create_actor(env_ptr, additional_assets[4], cube1_start_pose, "cube5", i, 2**6, 0)
            self._cube6_id = self.gym.create_actor(env_ptr, additional_assets[5], cube1_start_pose, "cube6", i, 2**7, 0)
            self._cube7_id = self.gym.create_actor(env_ptr, additional_assets[6], cube1_start_pose, "cube7", i, 2**8, 0)
            self._cube8_id = self.gym.create_actor(env_ptr, additional_assets[7], cube1_start_pose, "cube8", i, 2**9, 0)
            self._cube9_id = self.gym.create_actor(env_ptr, additional_assets[8], cube1_start_pose, "cube9", i, 2**10, 0)

            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self._cube1_id, 0, gymapi.MESH_VISUAL, colors[0])
            self.gym.set_rigid_body_color(env_ptr, self._cube2_id, 0, gymapi.MESH_VISUAL, colors[1])
            self.gym.set_rigid_body_color(env_ptr, self._cube3_id, 0, gymapi.MESH_VISUAL, colors[2])
            self.gym.set_rigid_body_color(env_ptr, self._cube4_id, 0, gymapi.MESH_VISUAL, colors[3])
            self.gym.set_rigid_body_color(env_ptr, self._cube5_id, 0, gymapi.MESH_VISUAL, colors[4])
            self.gym.set_rigid_body_color(env_ptr, self._cube6_id, 0, gymapi.MESH_VISUAL, colors[5])
            self.gym.set_rigid_body_color(env_ptr, self._cube7_id, 0, gymapi.MESH_VISUAL, colors[6])
            self.gym.set_rigid_body_color(env_ptr, self._cube8_id, 0, gymapi.MESH_VISUAL, colors[7])
            self.gym.set_rigid_body_color(env_ptr, self._cube9_id, 0, gymapi.MESH_VISUAL, colors[8])

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cube1_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cube2_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cube3_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cube4_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cube5_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cube6_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cube7_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cube8_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cube9_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()
        #duplicate heights and radius according to modulo applied above
        heights = self.cylinder_heights * (self.num_envs // NUM_OBJECTS)
        heights += self.cylinder_heights[:self.num_envs % NUM_OBJECTS]

        radius = self.cylinder_radius * (self.num_envs // NUM_OBJECTS)
        radius += self.cylinder_radius[:self.num_envs % NUM_OBJECTS]

        self.cylinder_heights = torch.tensor(heights, device = self.device)
        self.cylinder_radius = torch.tensor(radius, device = self.device)



    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        # self.handles = {
        #     "end_effector": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "gripper_ee"),
        #     "gripper_mover": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "gripper_mover"),
        #     "base_link" : self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "base_link"),
        #
        #     "link03" : self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "link03"),
        #     "link04" : self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "link04"),
        #     "link05" : self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "link05"),
        #     "link06" : self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "link06"),
        #
        #     "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
        # }

        self.handles = {
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "right_gripper_ee"),
            "base_link": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "base_link"),
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _force_sensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        self._force_sensor = gymtorch.wrap_tensor(_force_sensor).view(self.num_envs, -1)


        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]

        # self._eef_state = self._rigid_body_state[:, self.handles["end_effector"], :]
        # self._gripper_mover_state = self._rigid_body_state[:, self.handles["gripper_mover"], :]
        # self._base_link_state = self._rigid_body_state[:, self.handles["base_link"], :]
        #
        # self._link03_state = self._rigid_body_state[:, self.handles["link03"], :]
        # self._link04_state = self._rigid_body_state[:, self.handles["link04"], :]
        # self._link05_state = self._rigid_body_state[:, self.handles["link05"], :]
        # self._link06_state = self._rigid_body_state[:, self.handles["link06"], :]

        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._base_link_state = self._rigid_body_state[:, self.handles["base_link"], :]


        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cube1_state = self._root_state[:, self._cube1_id, :]
        self._cube2_state = self._root_state[:, self._cube2_id, :]
        self._cube3_state = self._root_state[:, self._cube3_id, :]
        self._cube4_state = self._root_state[:, self._cube4_id, :]
        self._cube5_state = self._root_state[:, self._cube5_id, :]
        self._cube6_state = self._root_state[:, self._cube6_id, :]
        self._cube7_state = self._root_state[:, self._cube7_id, :]
        self._cube8_state = self._root_state[:, self._cube8_id, :]
        self._cube9_state = self._root_state[:, self._cube9_id, :]


        # Initialize states
        # self.states.update({
        #     "cubeA_size": self.cylinder_heights / 2,
        #     "cube1_size": torch.ones_like(self._eef_state[:, 0]) * self.cube1_size,
        # })

        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            "cube1_size": torch.ones_like(self._eef_state[:, 0]) * self.cube1_size,
            "previous_q": torch.zeros((self.num_envs, self.num_dofs)).to(self.device),
            "previous_env_dt_q" : torch.zeros((self.num_envs, self.num_actions)).to(self.device),
            "previous_env_dt_q_full": torch.zeros((self.num_envs, self.num_actions)).to(self.device),
            "gripper_mean_torque": torch.zeros((self.num_envs, self.num_actions)).to(self.device),
            "prev_ee_pos" :  torch.zeros((self.num_envs, 3)).to(self.device),
            "prev_ee_quat": torch.zeros((self.num_envs, 4)).to(self.device),
            "cartesian_target" : torch.zeros((self.num_envs, 3)).to(self.device)
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self._arm_control = self._pos_control

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * (5 + 8), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        self.current_targets = torch.cat(
            (0.5 * torch.rand((self.num_envs, 1)), 0.5 * (torch.rand((self.num_envs, 1)) - 0.5),
             0.05 * torch.ones((self.num_envs, 1))), dim=1).to(self.device)

        self.current_cube_obs = torch.zeros((self.num_envs, 3), device = self.current_targets.device)
        self.current_cube1_obs = torch.zeros((self.num_envs, 3), device = self.current_targets.device)
        self.current_cube2_obs = torch.zeros((self.num_envs, 3), device = self.current_targets.device)
        self.current_cube3_obs = torch.zeros((self.num_envs, 3), device = self.current_targets.device)
        self.current_cube4_obs = torch.zeros((self.num_envs, 3), device = self.current_targets.device)
        self.current_cube5_obs = torch.zeros((self.num_envs, 3), device = self.current_targets.device)
        self.current_cube6_obs = torch.zeros((self.num_envs, 3), device = self.current_targets.device)
        self.current_cube7_obs = torch.zeros((self.num_envs, 3), device = self.current_targets.device)
        self.current_cube8_obs = torch.zeros((self.num_envs, 3), device = self.current_targets.device)
        self.current_cube9_obs = torch.zeros((self.num_envs, 3), device = self.current_targets.device)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_noise": self._q[:, :] + self.q_noise,

            "cartesian_targets" : self.cartesian_targets,
            # "default_dof_pos" : self.franka_default_dof_pos_2,
            "previous_action" : self.previous_actions,
            "mean_torque" : self.first_step_torques,
            "lower_limits" : self.franka_dof_lower_limits,
            "upper_limits" : self.franka_dof_upper_limits,
            "q_short": self._q[:, :8],

            "q_vel" : (self._q[:, :] - self.states['previous_q']) / self.sim_params.dt,
            "eef_pos_normalized": self._eef_state[:, :3] - self._base_link_state[:, :3],
            "eef_pos" : self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],

            "object_heights" : self.object_heights,

            # "gripper_pos_normalized": self._gripper_mover_state[:, :3] - self._base_link_state[:, :3],
            # "gripper_pos_normalized_noise": self._gripper_mover_state[:, :3] - self._base_link_state[:, :3] + self.gripper_noise,
            #
            # "gripper_mean_torque" : self.mean_gripper_torque,#self.first_step_torques[:, -1].unsqueeze(1), train
            #
            # "gripper_pos" : self._gripper_mover_state[:, :3],
            # "gripper_quat": self._gripper_mover_state[:, 3:7],
            # "gripper_vel": self._gripper_mover_state[:, 7:10],

            "target_normalized" : self.current_targets,

            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos_normalized": self._cubeA_state[:, :3] - self._base_link_state[:, :3],
            "cubeA_pos" : self._cubeA_state[:, :3],
            "cubeA_vel" : self._cubeA_state[:, 7:],

            "cube1_pos": self._cube1_state[:, :3] - self._base_link_state[:, :3],
            "cube2_pos": self._cube2_state[:, :3] - self._base_link_state[:, :3],
            "cube3_pos": self._cube3_state[:, :3] - self._base_link_state[:, :3],
            "cube4_pos": self._cube4_state[:, :3] - self._base_link_state[:, :3],
            "cube5_pos": self._cube5_state[:, :3] - self._base_link_state[:, :3],
            "cube6_pos": self._cube6_state[:, :3] - self._base_link_state[:, :3],
            "cube7_pos": self._cube7_state[:, :3] - self._base_link_state[:, :3],
            "cube8_pos": self._cube8_state[:, :3] - self._base_link_state[:, :3],
            "cube9_pos": self._cube9_state[:, :3] - self._base_link_state[:, :3],


            "cubeA_first_pos": self.current_cube_obs - self._base_link_state[:, :3],
            "cube1_first_pos": self.current_cube1_obs - self._base_link_state[:, :3],
            "cube2_first_pos": self.current_cube2_obs - self._base_link_state[:, :3],
            "cube3_first_pos": self.current_cube3_obs - self._base_link_state[:, :3],
            "cube4_first_pos": self.current_cube4_obs - self._base_link_state[:, :3],
            "cube5_first_pos": self.current_cube5_obs - self._base_link_state[:, :3],
            "cube6_first_pos": self.current_cube6_obs - self._base_link_state[:, :3],
            "cube7_first_pos": self.current_cube7_obs - self._base_link_state[:, :3],
            "cube8_first_pos": self.current_cube8_obs - self._base_link_state[:, :3],
            "cube9_first_pos": self.current_cube9_obs - self._base_link_state[:, :3],

            "to_target": (self._cubeA_state[:, :3] - self._base_link_state[:, :3]) - (self._eef_state[:, :3] - self._base_link_state[:, :3]),

            # "link03_pos" : self._link03_state[:, :3],
            # "link04_pos" : self._link04_state[:, :3],
            # "link05_pos" : self._link05_state[:, :3],
            # "link06_pos" : self._link06_state[:, :3],

        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        # Refresh states
        self._update_states()


    def compute_reward(self, actions):
        # print(self.torques)
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length, torch.tensor(self.running_epoch//32), self._force_sensor
        )

    def compute_observations(self):
        self._refresh()
        obs = ["q_short", "eef_pos_normalized", "cubeA_pos_normalized"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        #all_cubes = self.create_obstacles_tensor()
        # progress = self.progress_buf == 70
        # successful = self.states['cubeA_pos'][:, 2] > 1.07
        # successful = torch.logical_and(successful, progress)
        # with open('starting_points_290.txt', 'a+') as f:
        #     to_write = torch.cat((self.states['q'][successful], self.states['cubeA_pos'][successful], self.states['cubeA_quat'][successful]), dim = 1).tolist()
        #     for row in to_write:
        #         f.write(str(row) + '\n')
        return self.obs_buf


    def create_obstacles_tensor(self):
        all_cubes = torch.stack([self._cube1_state[:, :3], self._cube2_state[:, :3], self._cube3_state[:, :3], self._cube4_state[:, :3]], dim=0).permute(1, 0, 2)


        all_cubes = torch.cat([all_cubes, self.object_dims[:, 1:5, :]], dim=-1)
        all_cubes_on_table = all_cubes * (all_cubes[:,:,2]>1).unsqueeze(-1)

        return all_cubes_on_table


    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.current_targets[env_ids] = torch.cat((0.5 * torch.rand((len(env_ids), 1)), 0.5 * (torch.rand((len(env_ids), 1)) - 0.5),
                   0.15 * torch.ones((len(env_ids), 1))), dim=1).to(self.device)

        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=False)
        on_table_1 = self._reset_init_cube_state(cube='1', env_ids=env_ids, check_valid=False)
        on_table_2 = self._reset_init_cube_state(cube='2', env_ids=env_ids, check_valid=False)
        on_table_3 = self._reset_init_cube_state(cube='3', env_ids=env_ids, check_valid=False)
        on_table_4 = self._reset_init_cube_state(cube='4', env_ids=env_ids, check_valid=False)
        # on_table_5 = self._reset_init_cube_state(cube='5', env_ids=env_ids, check_valid=False)
        # on_table_6 = self._reset_init_cube_state(cube='6', env_ids=env_ids, check_valid=False)
        # on_table_7 = self._reset_init_cube_state(cube='7', env_ids=env_ids, check_valid=False)
        # on_table_8 = self._reset_init_cube_state(cube='8', env_ids=env_ids, check_valid=False)
        # on_table_9 = self._reset_init_cube_state(cube='9', env_ids=env_ids, check_valid=False)

        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]

        self._cube1_state[env_ids] = self._init_cube1_state[env_ids]
        self._cube2_state[env_ids] = self._init_cube2_state[env_ids]
        self._cube3_state[env_ids] = self._init_cube3_state[env_ids]
        self._cube4_state[env_ids] = self._init_cube4_state[env_ids]
        # self._cube5_state[env_ids] = self._init_cube5_state[env_ids]
        # self._cube6_state[env_ids] = self._init_cube6_state[env_ids]
        # self._cube7_state[env_ids] = self._init_cube7_state[env_ids]
        # self._cube8_state[env_ids] = self._init_cube8_state[env_ids]
        # self._cube9_state[env_ids] = self._init_cube9_state[env_ids]

        self.gripper_noise = 0. #* (torch.rand((self._gripper_mover_state[:, :3].shape), device=self._q.device) - 0.5)
        self.q_noise = 0. * (torch.rand((self._q.shape), device=self._q.device) - 0.5)

        reset_noise = torch.rand((len(env_ids), 11), device = self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
        # Overwrite gripper init pos (no noise since these are always position controlled)
        # print("THERE")
        # print(pos)

        self._q[env_ids, :] = pos

        self.actions[env_ids, :] = pos[:, :8]

        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        # self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        # self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._effort_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -10:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            other_cube_state = self._init_cubeA_state[env_ids, :]
            cube_heights = self.object_heights[env_ids, 0]
        elif cube.lower() == '1':
            this_cube_state_all = self._init_cube1_state
            other_cube_state = self._init_cube1_state[env_ids, :]
            cube_heights = self.object_heights[env_ids, 1]
        elif cube.lower() == '2':
            this_cube_state_all = self._init_cube2_state
            other_cube_state = self._init_cube2_state[env_ids, :]
            cube_heights = self.object_heights[env_ids, 2]
        elif cube.lower() == '3':
            this_cube_state_all = self._init_cube3_state
            other_cube_state = self._init_cube3_state[env_ids, :]
            cube_heights = self.object_heights[env_ids, 3]
        elif cube.lower() == '4':
            this_cube_state_all = self._init_cube4_state
            other_cube_state = self._init_cube4_state[env_ids, :]
            cube_heights = self.object_heights[env_ids, 4]
        elif cube.lower() == '5':
            this_cube_state_all = self._init_cube5_state
            other_cube_state = self._init_cube5_state[env_ids, :]
            cube_heights = self.object_heights[env_ids, 5]
        elif cube.lower() == '6':
            this_cube_state_all = self._init_cube6_state
            other_cube_state = self._init_cube6_state[env_ids, :]
            cube_heights = self.object_heights[env_ids, 6]
        elif cube.lower() == '7':
            this_cube_state_all = self._init_cube7_state
            other_cube_state = self._init_cube7_state[env_ids, :]
            cube_heights = self.object_heights[env_ids, 7]
        elif cube.lower() == '8':
            this_cube_state_all = self._init_cube8_state
            other_cube_state = self._init_cube8_state[env_ids, :]
            cube_heights = self.object_heights[env_ids, 8]
        elif cube.lower() == '9':
            this_cube_state_all = self._init_cube9_state
            other_cube_state = self._init_cube9_state[env_ids, :]
            cube_heights = self.object_heights[env_ids, 9]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Get correct references depending on which one was selected
        # if cube.lower() == 'a':
        #     this_cube_state_all = self._init_cubeA_state
        #     cube_heights = 0.05
        # elif cube.lower() == 'b':
        #     this_cube_state_all = self._init_cube1_state
        #     other_cube_state = self._init_cubeA_state[env_ids, :]
        #     cube_heights = self.states["cubeA_size"]
        # else:
        #     raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")
        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius

        min_dists = ((self.states["cubeA_size"] + self.states["cube1_size"])[env_ids] * np.sqrt(2) / 2.0)

        # We scale the min dist by 2 so that the cubes aren't too close together
        min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights / 2 + 0.001


        # def randomize_heading(cube_state):
        #     theta = torch.rand((cube_state.shape[0],)) * (2 * torch.pi) - torch.pi
        #     w = torch.cos(theta / 2)
        #     z = torch.sin(theta / 2)
        #     sampled_cube_state[:, 6] = w
        #     sampled_cube_state[:, 5] = z

        # randomize_heading(sampled_cube_state)
        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 3] = 0.0
        sampled_cube_state[:, 4] = 0.0
        sampled_cube_state[:, 5] = 0.0
        sampled_cube_state[:, 6] = 1.0
        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of fchecking based on cubes' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_cube_state[active_idx, :2] = centered_cube_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
                # Check if sampled values are valid
                cube_dist = torch.linalg                .norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample
            if cube.lower() == 'a':
                sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                                  2.0 * self.start_position_noise * (
                                                          torch.rand(num_resets, 2, device=self.device) - 0.5)
            else:
                sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                            2.0 * 0.3 * (
                                                    torch.rand(num_resets, 2, device=self.device) - 0.5)

        if cube.lower() != 'a':
            # on_table = torch.randint(2, size=(len(env_ids),)) == 1
            # sampled_cube_state[~on_table, 2] -= 1
            on_table = torch.randint(3, size=(len(env_ids),)) < 2
            sampled_cube_state[~on_table, 2] -= 1


        this_cube_state_all[env_ids, :] = sampled_cube_state
        if cube.lower() == 'a':
            self.current_cube_obs[env_ids] = sampled_cube_state[:, :3].clone()
            return None
        elif cube.lower() == '1':
            self.current_cube1_obs[env_ids] = sampled_cube_state[:, :3].clone()
        elif cube.lower() == '2':
            self.current_cube2_obs[env_ids] = sampled_cube_state[:, :3].clone()
        elif cube.lower() == '3':
            self.current_cube3_obs[env_ids] = sampled_cube_state[:, :3].clone()
        elif cube.lower() == '4':
            self.current_cube4_obs[env_ids] = sampled_cube_state[:, :3].clone()
        elif cube.lower() == '5':
            self.current_cube5_obs[env_ids] = sampled_cube_state[:, :3].clone()
        elif cube.lower() == '6':
            self.current_cube6_obs[env_ids] = sampled_cube_state[:, :3].clone()
        elif cube.lower() == '7':
            self.current_cube7_obs[env_ids] = sampled_cube_state[:, :3].clone()
        elif cube.lower() == '8':
            self.current_cube8_obs[env_ids] = sampled_cube_state[:, :3].clone()
        elif cube.lower() == '9':
            self.current_cube9_obs[env_ids] = sampled_cube_state[:, :3].clone()
        return on_table

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        action_tensor = actions
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        # mean_torques = torch.zeros((self.first_step_torques.shape[0], self.first_step_torques.shape[1]), device = self.first_step_torques.device)
        # mean_gripper_torque = torch.zeros((self.num_envs, 1), device = mean_torques.device)

        # self.states['previous_env_dt_q'] = self.states['q'][:, -1].clone().unsqueeze(1)
        # self.states['prev_ee_pos'] = self.states['eef_pos'][:, :].clone() - self._base_link_state[:, :3]
        # self.states['prev_ee_quat'] = self.states['eef_quat'][:, :].clone()

        for i in range(20):
            torques = self.compute_torques(self._pos_control)  # + self.additional_torques
            if i == 0:
                self.first_step_torques = torques.clone()
            upper_torque_limit = torch.tensor([10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5]).to(self.device)
            torques = torch.clip(torques, min=-upper_torque_limit, max=upper_torque_limit)
            # mean_torques += torques.abs()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques.detach()))
            self.states['previous_q'] = self.states['q'].clone()
            # mean_gripper_torque += torques[:, 6:]
            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self._update_states()
            if self.force_render:
                self.render()
        # if self.num_envs < 100:
        #     print(self.actions[-1])
        #     print(self.first_step_torques[-1])

        # mean_torques /= self.decimation
        # print(mean_torques.mean(dim=0))

        # self.mean_gripper_torque = mean_gripper_torque / self.decimation

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.depth_images = torch.zeros((self.num_envs, 2*64*64), device = self.device)
        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.extras['depth_images'] = self.depth_images.to(self.rl_device)
        self.extras['_cubeA_state'] =  self._cubeA_state[:, :7].to(self.rl_device)
        self.extras['obstacles'] = self.create_obstacles_tensor()

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def compute_torques(self, actions):
        kp = torch.tensor([100., 100., 65., 100., 22.78, 49.521, 10., 10., 10., 10., 10.]).to(self.device)
        kd = torch.tensor([50., 50., 65., 50., 0.671, 2.199, 0.15, 0.001, 0.001, 0.001, 0.001]).to(self.device)
        torques = kp * (actions - self.states['q']) - kd * self.states['q_vel']
        return torques

    def euler_to_quaternion(self, roll, pitch, yaw):
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return w, x, y, z

    def pre_physics_step(self, actions):
        # print(self.states['cubeA_pos_normalized'][-1])
        # print('____')
        # print(self._q[-1])
        # print('____')

        actual_actions = torch.zeros((self.actions.shape[0], 11), device = self.actions.device)
        computed_actions = self._pos_control[:, :7].clone() + (1/200) * actions[:, :-1] * 7.5

        last_action_positive = actions[:, -1] > 0
        last_action_negative = actions[:, -1] < 0

        actual_actions[:, :7] = computed_actions[:, :7].clone()

        actual_actions[last_action_positive, 7] = 0.87
        actual_actions[last_action_positive, 8] = 0.87
        actual_actions[last_action_positive, 9] = 0.87
        actual_actions[last_action_positive, 10] = 0.87

        actual_actions[last_action_negative, 7] = 0
        actual_actions[last_action_negative, 8] = 0
        actual_actions[last_action_negative, 9] = 0
        actual_actions[last_action_negative, 10] = 0

        actual_actions = actual_actions.clip(self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self._pos_control = actual_actions


    def transform_to_base_frame(self,
                                target_ee_frame_pos,
                                target_ee_frame_quat,
                                prev_base_frame_ee_pos,
                                prev_base_frame_ee_quat):
        target_base_frame_pos = my_quat_rotate(prev_base_frame_ee_quat, target_ee_frame_pos) + prev_base_frame_ee_pos
        target_base_frame_quat = quat_mul(target_ee_frame_quat, prev_base_frame_ee_quat)
        return target_base_frame_pos, target_base_frame_quat



    def post_physics_step(self):
        self.progress_buf += 1
        self.running_epoch += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # if  self.current_episode % 20 == 0 and self.decimation<20 and :
        #     self.decimation += 1
        #     print('decimation up: ', self.decimation)

        # self.current_episode = torch.tensor(self.running_epoch//32)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.previous_actions = self.actions.clone()

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cube1_pos = self.states["cube1_pos"]
            cube1_rot = self.states["cube1_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos, cube1_pos), (eef_rot, cubeA_rot, cube1_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length, current_episode, force_sensor
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    

    # Compute per-env physical parameters
    target_height = states["cubeA_size"] / 2.0
    # cubeA_size = states["cubeA_size"]
    # cube1_size = states["cube1_size"]
    # (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]
    d = torch.norm(states["cubeA_pos"] - states["eef_pos"], dim=-1)
    # print(d)
    # close_to_goal = d < 0.01
    close_to_goal = d < 0.005

    dist_reward = 1 - torch.tanh((5 * d))
    table_h = reward_settings["table_height"]
    cubeA_height = states["cubeA_pos"][:, 2] - table_h
    cubeA_lifted = (cubeA_height) > 0.06

    # positive_torque = states['mean_torque'][:, -1] > 2 #works with mean torques
    # gripper_stuck = torch.abs((states['q'][:, -1] - states['previous_env_dt_q'].squeeze(1))) < 0.01
    # contact_reward = positive_torque * gripper_stuck * close_to_goal
    # contact_reward = positive_torque * gripper_stuck * close_to_goal
    # from_side = dist_reward * torch.exp(- from_side - from_side_b)
    #
    # dist_from_target = torch.exp(-5*torch.norm(states['target_normalized'] - states['cubeA_pos_normalized'], dim = -1))
    # bad_coordinate = ((states['cartesian_targets'][:, 2] + states['gripper_pos_normalized'][:, 2]) < 0.03).long()
    # print(states['target_normalized'])

    # no_rotation = torch.abs(states['q'][:, -2])
    # obstacle_fell = states['cube1_first_pos'][:, 2] < 0.10.
    close_hand = torch.exp(-torch.norm(states['q'][:, -4:], dim=-1))
    open_hand = torch.norm(states['q'][:, -4:], dim=-1)

    # distance_to_final_target = torch.exp(-10 * torch.norm(states['cubeA_pos'] - states['pick_target'], dim=-1))
    rewards = 1 * dist_reward + 10 * cubeA_lifted + close_to_goal * close_hand #+ cubeA_lifted #* distance_to_final_target * 5  # + close_to_goal * #+ open_hands#+ 1 * contact_reward + 5 * contact_reward * dist_from_target #- obstacle_fell.long()  #- 5 * bad_coordinate#+ from_side #+ #+ torque_reward + action_diff #+ action_diff #no_rotation

    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
    cube_fall = states['cubeA_pos'][:, 2] < 1
    reset_buf |= cube_fall
    reset_buf |= (cubeA_height) > 0.5

    #
    # reset_buf |= (states['cubeA_pos'][:, 0] < -0.40)
    #reset_buf |= (states['cubeA_pos'][:, 0] > 0.15)
    #reset_buf |= (states['cubeA_pos'][:, 1] < -0.6)
    #reset_buf |= (states['cubeA_pos'][:, 1] > 0.6)

    # reset_buf |= torch.any((states['q'] > states['upper_limits']) | (states['q'] < states['lower_limits']), dim=1)

    # TERMINATION ON HEIGHT CHANGE OF NON TARGET OBJECTS
    reset_buf |= ((states['cube1_pos'][:, 2] < (states['object_heights'][:, 1]/2 - 0.025)) & (states['cube1_first_pos'][:, 2] > 0) ) #/ 2
    reset_buf |= ((states['cube2_pos'][:, 2] < (states['object_heights'][:, 2]/2 - 0.025)) & (states['cube2_first_pos'][:, 2] > 0) ) #/ 2
    reset_buf |= ((states['cube3_pos'][:, 2] < (states['object_heights'][:, 3]/2 - 0.025)) & (states['cube3_first_pos'][:, 2] > 0) ) #/ 2
    reset_buf |= ((states['cube4_pos'][:, 2] < (states['object_heights'][:, 4]/2 - 0.025)) & (states['cube4_first_pos'][:, 2] > 0) ) #/ 2


    #TERMINATION ON XY CHANGE OF NON TARGET OBJECTS
    reset_buf |= (torch.norm(states['cube1_pos'][:, :2] - states['cube1_first_pos'][:, :2], dim = -1) > 0.02) & (states['cube1_first_pos'][:, 2] > 0)
    reset_buf |= (torch.norm(states['cube2_pos'][:, :2] - states['cube2_first_pos'][:, :2], dim = -1) > 0.02) & (states['cube2_first_pos'][:, 2] > 0)
    reset_buf |= (torch.norm(states['cube3_pos'][:, :2] - states['cube3_first_pos'][:, :2], dim = -1) > 0.02) & (states['cube3_first_pos'][:, 2] > 0)
    reset_buf |= (torch.norm(states['cube4_pos'][:, :2] - states['cube4_first_pos'][:, :2], dim = -1) > 0.02) & (states['cube4_first_pos'][:, 2] > 0)



    # print(states['cube1_pos'][:, 2])
    # print(states['cube1_first_pos'][:, 2])
    # print(torch.norm(states['eef_vel'][:, :3], dim = -1))
    #TODO reset on too high torques
    # print(states['mean_torque'].mean(dim = 0))
    return rewards, reset_buf

