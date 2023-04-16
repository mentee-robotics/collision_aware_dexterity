import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym.torch_utils import *
#from Pointnet_Pointnet2_pytorch import my_pointnet
import sys

# p = "/home/guy/VSprojects/pointcloud_dexterity/Pointnet_Pointnet2_pytorch/visualizer"
# sys.path.append(p) if p not in sys.path else None
# from Pointnet_Pointnet2_pytorch.visualizer.pc_utils import draw_point_cloud
from typing import Dict, Any, Tuple

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
        assert self.control_type in {"osc", "joint_tor"}, \
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)

        ## FOR VISUALS:
        self.use_pointcloud = False
        self.use_depth_sensors = True

        self.use_autoencoder = True
        self.use_pretrained_autoencoder = False
        self.train_autoencoder = False

        ## WITHOUT VISUALS:
        # self.use_pointcloud = False
        # self.use_depth_sensors = False
        # self.use_autoencoder = True
        # self.use_pretrained_autoencoder = False
        # self.train_autoencoder = False

        self.z_size = 32

        assert not (
                    self.use_pointcloud and self.use_depth_sensors), "work only in one of the two modes: use_pointcloud or use_depth_sensors"
        if self.use_pointcloud:
            pointcloud_or_depth_embeddings = 1024
            self.process_visual_every = '% 20'
        if self.use_depth_sensors:
            pointcloud_or_depth_embeddings = 32 * 32  # 128*128
            self.process_visual_every = '% 1'

        pointcloud_or_depth_embeddings = pointcloud_or_depth_embeddings if not self.use_autoencoder else self.z_size

        # n_obs = 17 + 1024 if self.use_pointcloud else 17
        # self.dummy_project = torch.nn.Linear(1024,10, bias=False)
        n_obs = 17 + pointcloud_or_depth_embeddings if (self.use_pointcloud or self.use_depth_sensors) else 17
        self.cfg["env"]["numObservations"] = 15 + self.z_size #n_obs - 3 + 1

        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7

        # Values to be filled in at runtime
        self.states = {}  # will be dict filled with relevant states to use for reward calculation
        self.handles = {}  # will be dict mapping names to relevant sim handles
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed
        self._init_cubeA_state = None  # Initial state of cubeA for the current env
        self._init_cubeB_state = None  # Initial state of cubeB for the current env
        self._cubeA_state = None  # Current state of cubeA for the current env
        self._cubeB_state = None  # Current state of cubeB for the current env
        self._cubeA_id = None  # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None  # Actor ID corresponding to cubeB for a given env

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None  # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._franka_effort_limits = None  # Actuator effort limits for franka
        self._global_indices = None  # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.actor_handles = []

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.actions = torch.zeros((self.num_envs, self.num_actions)).cuda()
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions)).cuda()
        self.previous_eef_pos = torch.zeros((self.num_envs, 3)).cuda()
        self.previous_cubeA_pos = torch.zeros((self.num_envs, 3)).cuda()
        self.torques = torch.zeros((self.num_envs, self.num_actions)).cuda()
        self.test_mode = self.num_envs < 100

        self.franka_default_dof_pos = to_torch(
            [0, 0, 0, -1.3, 0, 0., -0.7], device=self.device
        )
        #[0, 1.55, -2.0, 1.0, 0, 0, -0.7]

        # Set control limits
        self.cmd_limit = self._franka_effort_limits[:].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.visual_features = torch.zeros((self.num_envs, pointcloud_or_depth_embeddings), device=self.device)
        # self.depth_images = torch.zeros((self.num_envs, 128, 180), device = self.device)

        # Refresh tensors
        self._refresh()

    def create_depth_sensors(self):
        # Depth camera sensor
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = 60  # 55  # 90 #50 #120
        # camera_props.near_plane = 75
        camera_props.width = 180  # 10 #64
        camera_props.height = 128  # 10  #64
        self.depth_map_size = camera_props.width * camera_props.height

        camera_props.enable_tensors = True

        # local_transform = gymapi.Transform()
        # local_transform.p = gymapi.Vec3(0.3, 0, 0.7)  # (-0.026, 0, -0,027)
        # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(90))  # 45.0))

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0.1, 0, 0.15)  # (-0.026, 0, -0,027)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(30))  # 45.0))

        # local_transform = gymapi.Transform()
        # local_transform.p = gymapi.Vec3(0.18, 0.0, -0.04)  # (-0.026, 0, -0,027)
        # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(0))  # 45.0))

        # local_transform = gymapi.Transform()
        # local_transform.p = gymapi.Vec3(0.65, 0, 0.3)  # (-0.026, 0, -0,027)
        # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.45, 0, 1.0), np.radians(-180))  # 45.0))



        ########## RGB camera sensor ############
        camera_props_rgb = gymapi.CameraProperties()
        camera_props_rgb.horizontal_fov = 95  # 55  # 90 #50 #120
        # camera_props.near_plane = 75
        camera_props_rgb.width = 128  # 10 #64
        camera_props_rgb.height = 128  # 10  #64
        self.depth_map_size_rgb = camera_props_rgb.width * camera_props_rgb.height

        camera_props_rgb.enable_tensors = True

        # local_transform_rgb = gymapi.Transform()
        # local_transform_rgb.p = gymapi.Vec3(0.3, 0, 0.7)  # (-0.026, 0, -0,027)
        # local_transform_rgb.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(90))  # 45.0))

        local_transform_rgb = gymapi.Transform()
        local_transform_rgb.p = gymapi.Vec3(0.0, 0.25, -0.1)   #gymapi.Vec3(0.05, 0, 0.3)    # (-0.026, 0, -0,027)  (0.1, 0, 0.15)
        local_transform_rgb.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, -1, 0), np.radians(60))  # gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(70))  # 45.0))

        # local_transform_rgb = gymapi.Transform()
        # local_transform_rgb.p = gymapi.Vec3(0.18, 0.0, -0.04)  # (-0.026, 0, -0,027)
        # local_transform_rgb.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(0))  # 45.0))

        # local_transform_rgb = gymapi.Transform()
        # local_transform_rgb.p = gymapi.Vec3(0.65, 0, 0.3)  # (-0.026, 0, -0,027)
        # local_transform_rgb.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.45, 0, 1.0), np.radians(-180))  # 45.0))


        self.cameras = []
        self.cameras_rgb = []
        for i in range(self.num_envs):
            env_handler = self.envs[i]
            humanoid_handler = self.actor_handles[i]
            body_handler = self.gym.find_actor_rigid_body_handle(env_handler, humanoid_handler, "link06")
            body_handler_rgb = self.gym.find_actor_rigid_body_handle(env_handler, humanoid_handler, "base_link")

            camera_handler = self.gym.create_camera_sensor(env_handler, camera_props)
            camera_handler_rgb = self.gym.create_camera_sensor(env_handler, camera_props_rgb)

            self.gym.attach_camera_to_body(
                camera_handler, env_handler, body_handler, local_transform, gymapi.FOLLOW_TRANSFORM
            )
            self.gym.attach_camera_to_body(
                camera_handler_rgb, env_handler, body_handler_rgb, local_transform_rgb, gymapi.FOLLOW_TRANSFORM
            )

            self.cameras.append(camera_handler)
            self.cameras_rgb.append(camera_handler_rgb)

    def create_depth_sensors_buffer(self):
        # Camera sensors
        depth_sensors = []
        segmentation_sensors = []
        rgb_sensors = []
        segmentation_rgb_sensors = []
        for i in range(self.num_envs):
            env_handler = self.envs[i]
            cam_handler = self.cameras[i]
            cam_handler_rgb = self.cameras_rgb[i]
            camera_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handler, cam_handler, gymapi.IMAGE_SEGMENTATION  # IMAGE_COLOR#IMAGE_DEPTH
            )

            camera_tensor_d = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handler, cam_handler, gymapi.IMAGE_DEPTH  # IMAGE_COLOR#IMAGE_DEPTH
            )

            # camera_tensor_rgb_segmented = self.gym.get_camera_image_gpu_tensor(
            #     self.sim, env_handler, cam_handler_rgb, gymapi.IMAGE_SEGMENTATION  # IMAGE_COLOR#IMAGE_DEPTH
            # )

            camera_tensor_rgb = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handler, cam_handler_rgb, gymapi.IMAGE_COLOR  # IMAGE_COLOR#IMAGE_DEPTH
            )

            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            torch_camera_tensor_d = gymtorch.wrap_tensor(camera_tensor_d)
            # torch_camera_tensor_rgb_seg = gymtorch.wrap_tensor(camera_tensor_rgb_segmented)
            torch_camera_tensor_rgb = gymtorch.wrap_tensor(camera_tensor_rgb)
            segmentation_sensors.append(torch_camera_tensor)
            depth_sensors.append(torch_camera_tensor_d)
            rgb_sensors.append(torch_camera_tensor_rgb)
            # segmentation_rgb_sensors.append(torch_camera_tensor_rgb_seg)

        self.depth_sensors = depth_sensors
        self.segmentation_sensors = segmentation_sensors
        self.rgb_sensors = rgb_sensors
        # self.segmentation_rgb_sensors = segmentation_rgb_sensors

    def get_depth_sensors(self):
        # self.s
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        depth_sensors = torch.stack(self.depth_sensors)
        segmentation_sensors = torch.stack(self.segmentation_sensors)
        rgb_sensors = torch.stack(self.rgb_sensors)
        #segmentation_rgb_sensors = torch.stack(self.segmentation_rgb_sensors)
        
        rgb_sensors = rgb_sensors.permute(0, 3, 1, 2).float() / 255.
        rgb_sensors = rgb_sensors[:,:3,:,:]

        #print(depth_sensors)
        rgb_sensors = torch.clip(rgb_sensors, min=-5, max=5)
        depth_sensors = torch.clip(depth_sensors, min=-5, max=5)
        #depth_sensors = (depth_sensors - depth_sensors.mean())/depth_sensors.std()

        #self.is_cube_in_frame = ((segmentation_sensors.view(segmentation_sensors.shape[0], -1) == 2).sum(-1) > 0).int()
        self._segmentation_sensors = segmentation_sensors


        self.gym.end_access_image_tensors(self.sim)
        mask = (segmentation_sensors == 1) | (segmentation_sensors == 2) # franka or cube
        # mask_rgb = (segmentation_rgb_sensors == 1) | (segmentation_rgb_sensors == 2) # franka or cube


        return depth_sensors * mask, rgb_sensors #*mask_rgb

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.use_depth_sensors:
            self.create_depth_sensors()
            self.create_depth_sensors_buffer()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "z1_pro/urdf/z1_limited.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        # asset_options.thickness = 0.001
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.default_dof_drive_mode = 3  # effort mode
        asset_options.use_mesh_materials = True
        asset_options.vhacd_enabled = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # stiffness = [0., 0., 0., 0., 0., 0., 0]
        # damping = [0., 0., 0., 0., 0., 0., 0]
        # stiffness = [30., 30., 30., 30., 30., 30., 0]
        # damping = [1., 1., 1., 1., 1., 1., 0]

        stiffness = [0.0] * 7
        armature = [0.001, 0.03, 0.181, 0.153, 0.138, 0.007, 0.001]
        damping = [0.722, 7.596, 11.769, 5.423, 2.651, 1.645, 0.859]
        friction = [0.093, 0.017, 0.034, 0.04, 0.203, 0.444, 0.58]

        # Create table asset
        table_pos = [-0.15, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.5, 0.7, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.01, 0.01, table_stand_height], table_opts)

        self.cubeA_size = 0.04
        self.cubeB_size = 0.05

        # cubeB_asset = self.gym.load_asset(self.sim, '', "/home/raphael/media/bensadoun/IsaacGymEnvs_v4_multicube/assets/urdf/objects/tray/urdf/tray.urdf", bottle_opts)
        # cubeB_color = gymapi.Vec3(1., 1., 1.)

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
        for i in range(self.num_franka_dofs):
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

        # TODO wtf
        # self.franka_dof_speed_scales[[7, 8]] = 0.1
        # franka_dof_props['effort'][7] = 200
        # franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, -0.25, 1.38) #gymapi.Vec3(-0.45, 0.0, 1.0)
        franka_start_pose.r = gymapi.Quat(0, -0.707, 0, -0.707) #gymapi.Quat(-0.7071, 0, -0.7071, 0) # #gymapi.Quat(0.0, 0.0, 0.0, 1.0)

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
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        tray_stand_start_pose = gymapi.Transform()
        tray_stand_start_pose.p = gymapi.Vec3(*[-0.35, 0., 1.03])
        tray_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4  # 1 for table, table stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 4  # 1 for table, table stand, cubeA, cubeB

        self.frankas = []
        self.envs = []

        NUM_OBJECTS = 20
        self.running_epoch = 0
        self.current_episode = 0
        cubeA_assets = []
        import open3d as o3d
        # point_clouds = []
        # for i in range(NUM_OBJECTS):
        #     point_cloud_cylinder = o3d.io.read_point_cloud(f"../point_clouds/cylinders/cylinder_{i}.pcd")
        #     point_cloud_box = o3d.io.read_point_cloud(f"../point_clouds/box/box_{i}.pcd")
        #     point_clouds.append(point_cloud_cylinder)
        #     point_clouds.append(point_cloud_box)
        bottle_opts = gymapi.AssetOptions()
        bottle_opts.vhacd_enabled = True
        bottle_opts.vhacd_params.resolution = 300000
        bottle_opts.vhacd_params.max_convex_hulls = 50
        bottle_opts.vhacd_params.max_num_vertices_per_ch = 64
        self.cylinder_heights = []
        self.cylinder_radius = []
        self.object_heights = []

        self.cube_heights = []
        for i in range(NUM_OBJECTS):
            print(f"Loading urdf {i}")
            #cubeA_asset = self.gym.load_asset(self.sim, '../urdfs/cylinders', f"cylinder_{i}.urdf", bottle_opts)
            cubeA_asset = self.gym.load_asset(self.sim, '/home/guy/VSprojects/learnable_encoding/assets/PLATE_NEW/urdf', "plate.urdf", bottle_opts)
            # with open(f'../meshes/cylinders/cylinder_{i}.txt', 'r') as f:
            #     line = f.readline()
            # line = line.strip().split(',')
            #self.cylinder_heights.append(float(line[0]))
            # self.cylinder_radius.append(float(line[1]))
            cubeA_assets.append(cubeA_asset)

            #self.object_heights.append(float(line[0]))
            self.object_heights.append(0.01)

            print(f"Loading urdf {i}")
            #cubeA_asset = self.gym.load_asset(self.sim, '../urdfs/box', f"box_{i}.urdf", bottle_opts)
            cubeA_asset = self.gym.load_asset(self.sim, '/home/guy/VSprojects/learnable_encoding/assets/PLATE_NEW/urdf', "plate.urdf", bottle_opts)

            # with open(f'../meshes/box/box_{i}.txt', 'r') as f:
            #     line = f.readline()
            # line = line.strip().split(',')
            # self.cube_heights.append(float(line[2]))

            self.cube_heights.append(0.01)
            cubeA_assets.append(cubeA_asset)

            #self.object_heights.append(float(line[2]))
            self.object_heights.append(0.01)

        # Create environments
        for i in range(self.num_envs):

            cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

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
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 1)
            self.actor_handles.append(franka_actor)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

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
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_assets[i % (NUM_OBJECTS * 2)], cubeA_start_pose,
                                                   "cubeA", i, 4, 2)
            # self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            # self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._cubeA_state_steps = torch.zeros(self.num_envs, 3, device=self.device)
        # self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # duplicate heights and radius according to modulo applied above
        # heights = self.cylinder_heights * (self.num_envs // NUM_OBJECTS)
        # heights += self.cylinder_heights[:self.num_envs % NUM_OBJECTS]
        #
        # radius = self.cylinder_radius * (self.num_envs // NUM_OBJECTS)
        # radius += self.cylinder_radius[:self.num_envs % NUM_OBJECTS]

        all_heights = self.object_heights * (self.num_envs // (NUM_OBJECTS * 2))
        all_heights += self.object_heights[: self.num_envs % (NUM_OBJECTS * 2)]

        # self.point_clouds = {i: point_clouds[i % (NUM_OBJECTS * 2)] for i in range(self.num_envs)}

        self.first_iteration = True

        self.object_heights = torch.tensor(all_heights, device=self.device)

        # Setup data
        self.init_data()

    def oversample(self, pointcloud, num_points):
        oversample_indices = torch.multinomial(torch.tensor([i for i in range(len(pointcloud))]).float(),
                                               num_points - len(pointcloud), replacement=True)
        pointcloud = torch.cat((pointcloud, pointcloud[oversample_indices]), dim=0)
        return pointcloud

    def undersample(self, pointcloud, num_points):
        undersample_indices = torch.multinomial(torch.tensor([i for i in range(len(pointcloud))]).float(), num_points,
                                                replacement=True)
        pointcloud = pointcloud[undersample_indices]
        return pointcloud

    def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        i, j, k, r = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def normalize_pointcloud(self, points):
        centered_points = points - points.mean(dim=0)
        max_dist = torch.max(torch.norm(centered_points, dim=-1, keepdim=True), dim=1, keepdim=True)[0].repeat(1,
                                                                                                               centered_points.shape[
                                                                                                                   1],
                                                                                                               centered_points.shape[
                                                                                                                   2])
        centered_points = centered_points / max_dist
        return centered_points

    def project_pointcloud(self, env_ids, cube_quats):
        projected_pointclouds = torch.zeros((len(env_ids), 512, 3))
        euler_quats = self.quaternion_to_matrix(cube_quats)
        for i in range(len(env_ids)):
            pointcloud = self.point_clouds[env_ids[i].item()]
            pointcloud = pointcloud.rotate(np.array(euler_quats[i].cpu().numpy()))
            diameter = np.linalg.norm(np.asarray(pointcloud.get_max_bound()) - np.asarray(pointcloud.get_min_bound()))
            camera = [0.5, 0, diameter]
            radius = diameter * 100
            _, pt_map = pointcloud.hidden_point_removal(camera, radius)
            proj_pointcloud = torch.tensor(np.array(pointcloud.points))[pt_map]
            if proj_pointcloud.shape[0] < self.POINTCLOUD_NUM_POINTS:
                proj_pointcloud = self.oversample(proj_pointcloud, self.POINTCLOUD_NUM_POINTS)
            else:
                proj_pointcloud = self.undersample(proj_pointcloud, self.POINTCLOUD_NUM_POINTS)
            projected_pointclouds[i] = proj_pointcloud
        projected_pointclouds = self.normalize_pointcloud(projected_pointclouds)
        return projected_pointclouds

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            "end_effector": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "gripper_ee"),
            "gripper_mover": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "gripper_mover"),
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            "base_link": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "base_link"),

            "link03": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "link03"),
            "link04": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "link04"),
            "link05": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "link05"),
            "link06": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "link06"),

        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print(self.num_dofs)

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]

        self._eef_state = self._rigid_body_state[:, self.handles["end_effector"], :]
        self._gripper_mover_state = self._rigid_body_state[:, self.handles["gripper_mover"], :]
        self._base_link_state = self._rigid_body_state[:, self.handles["base_link"], :]

        self._link03_state = self._rigid_body_state[:, self.handles["link03"], :]
        self._link04_state = self._rigid_body_state[:, self.handles["link04"], :]
        self._link05_state = self._rigid_body_state[:, self.handles["link05"], :]
        self._link06_state = self._rigid_body_state[:, self.handles["link06"], :]

        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # Initialize states
        self.states.update({
            "cubeA_size": self.object_heights / 2,
            "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
            "previous_q": torch.zeros((self.num_envs, self.num_actions)).cuda()

        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self._arm_control = self._pos_control

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "noised_q": self._q[:, :] + self._noise_for_q,
            "q_vel": (self._q[:, :] - self.states['previous_q']) / self.sim_params.dt,

            "eef_pos_normalized": self._eef_state[:, :3] - self._base_link_state[:, :3],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "previous_eef_pos": self.previous_eef_pos,

            "previous_action": self.previous_actions,
            "mean_torque": self.torques,
            "gripper_torque": self.torques[:, -1][:, None],

            "gripper_pos_normalized": self._gripper_mover_state[:, :3] - self._base_link_state[:, :3],
            "noised_gripper_pos_normalized": self._gripper_mover_state[:, :3] - self._base_link_state[:,
                                                                                :3] + self._noise_gripper_pos,
            "gripper_pos": self._gripper_mover_state[:, :3],
            "gripper_quat": self._gripper_mover_state[:, 3:7],
            "gripper_vel": self._gripper_mover_state[:, 7:10],

            "in_gripper_pos": (self._gripper_mover_state[:, :3] + self._eef_state[:, :3])/2,

            "link03_pos": self._link03_state[:, :3],
            "link04_pos": self._link04_state[:, :3],
            "link05_pos": self._link05_state[:, :3],
            "link06_pos": self._link06_state[:, :3],

            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos_normalized": self._cubeA_state[:, :3] - self._base_link_state[:, :3],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_vel": self._cubeA_state[:, 7:],
            "previous_cubeA_pos": self.previous_cubeA_pos,
            "cubeA_init_pos": self._cubeA_state_steps,

            # "to_target": self._cubeA_state[:, :3] - self._gripper_mover_state[:, :3],
            "to_target": (self._cubeA_state[:, :3] - self._base_link_state[:, :3]) - (
                        self._gripper_mover_state[:, :3] - self._base_link_state[:, :3]),

            "visual_features": self.visual_features

        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length,
            self.current_episode, self._segmentation_sensors
        )

    def compute_observations(self):
        self._refresh()
        # q = "q" if self.test_mode else "noised_q"
        obs = ["q", "gripper_pos_normalized", "gripper_quat", "gripper_torque"]  # "q", "gripper_pos_normalized", "gripper_quat", , "to_target"
        # if self.use_pointcloud or self.use_depth_sensors:
        #     obs += ["visual_features"]

        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        if self.use_pointcloud or self.use_depth_sensors:
            if not self.first_iteration:
                first_step_ids = torch.where(eval(f"self.progress_buf {self.process_visual_every} == 0"))[
                    0]  # % self.process_visual_every
            else:
                first_step_ids = torch.where(self.progress_buf == 1)[0]
                self.first_iteration = False
            if len(first_step_ids) > 0:
                if self.use_pointcloud:
                    projected_pointclouds = self.project_pointcloud(first_step_ids,
                                                                    self.states['cubeA_quat'][first_step_ids]).cuda()

                    BATCH_SIZE = 256 if not self.test_mode else self.progress_buf.shape[0]
                    for i in range(len(first_step_ids) // BATCH_SIZE):
                        projected_pointclouds_features = self.point_net(
                            projected_pointclouds[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE].permute(0, 2, 1))

                        with torch.set_grad_enabled(self.train_autoencoder):
                            self.visual_features[first_step_ids[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE]] = \
                                self.autoencoder.train_iter(
                                    projected_pointclouds_features[1].squeeze(-1)) if self.use_autoencoder else \
                                projected_pointclouds_features[1].squeeze(-1)
                    i = len(first_step_ids) // BATCH_SIZE - 1
                    if len(first_step_ids) % BATCH_SIZE != 0 and i != -1:
                        projected_pointclouds_features = self.point_net(
                            projected_pointclouds[i * BATCH_SIZE + BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE + (
                                    len(first_step_ids) % BATCH_SIZE)].permute(0, 2, 1))

                        with torch.set_grad_enabled(self.train_autoencoder):
                            self.visual_features[first_step_ids[
                                                 i * BATCH_SIZE + BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE + (
                                                         len(first_step_ids) % BATCH_SIZE)]] = self.autoencoder.train_iter(
                                projected_pointclouds_features[1].squeeze(-1)) if self.use_autoencoder else \
                            projected_pointclouds_features[1].squeeze(-1)
                
                elif self.use_depth_sensors:
                    depth_obs, rgb_obs = self.get_depth_sensors()
                    depth_obs = torch.clip(depth_obs, min=-5, max=5)
                    depth_images = torchvision.transforms.functional.resize(depth_obs, (64, 64)).view(-1,1, 64 * 64)
                    rgb_images = torchvision.transforms.functional.resize(rgb_obs, (64, 64)).view(-1, 3, 64 * 64) #.view(-1, 3, 64 * 64)
                    self.depth_images = torch.cat((depth_images, rgb_images), dim=1).view(-1, 4 * 64 * 64)

                    if self.test_mode and False:
                        for ii in range(1,3):
                            dd = depth_obs[-ii]
                            p = plt.imshow(dd.cpu().numpy())
                            #plt.savefig(f'./depth_natural_{self.progress_buf[-1]}.png')
                            plt.savefig(f'./depth_natural_{ii}.png')



                    # dd = torchvision.transforms.functional.resize(depth_obs, (32,32))[-1]
                    # p = plt.imshow(dd.cpu().numpy())
                    # plt.savefig('./depth_natural_mini.png')

                    # depth_obs = depth_obs.flatten(1, 2).cuda()
                    # with torch.set_grad_enabled(self.train_autoencoder):
                    #     # , self._cubeA_state
                    #     # , self._cubeA_state[:,:7]
                    #     self.visual_features[first_step_ids] = \
                    #     self.autoencoder.train_iter(depth_obs.squeeze(-1), self._cubeA_state[:, :7])[
                    #         first_step_ids] if self.use_autoencoder else depth_obs[
                    #         first_step_ids]  # depth_obs[1].squeeze(-1)[first_step_ids]
                        # all_indices = torch.arange(self.visual_features.shape[0], dtype=torch.long).to(self.visual_features.device)
                        # non_active = all_indices[~torch.isin(all_indices, first_step_ids)]
                        # self.visual_features[non_active] = torch.zeros_like(self.visual_features[non_active])

        # if self.use_depth_sensors:
        #     depth_obs = self.get_depth_sensors()
        #     depth_encoding = self.depth_encoder(depth_obs.flatten(1,2).cuda())
        #     self.obs_buf = torch.cat((self.obs_buf, depth_encoding), dim = -1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=False)

        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeA_state_steps[env_ids] = self._init_cubeA_state[env_ids][:, :3].clone()

        reset_noise = torch.rand((len(env_ids), 7), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
        # Overwrite gripper init pos (no noise since these are always position controlled)

        # pos[:, DOF_LEFT_ELBOW] = 2
        # pos[:, DOF_RIGHT_ELBOW_PITCH] = -2
        # Reset the internal obs accordingly

        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        self._pos_control[env_ids, :] = pos
        self.actions[env_ids, :] = pos

        self._noise_for_q = 0.05 * torch.randn_like(self._q[:, :])
        self._noise_gripper_pos = 0.01 * torch.randn_like(self._q[:, :3])

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1:].flatten()
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

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            # other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.object_heights
        elif cube.lower() == 'b':
            this_cube_state_all = self._init_cubeB_state
            other_cube_state = self._init_cubeA_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        min_dists = ((self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0)

        # We scale the min dist by 2 so that the cubes aren't too close together
        min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2 + 0.01

        def randomize_heading(cube_state):
            theta = torch.rand((cube_state.shape[0],)) * (2 * torch.pi) - torch.pi
            w = torch.cos(theta / 2)
            z = torch.sin(theta / 2)
            sampled_cube_state[:, 6] = w
            sampled_cube_state[:, 5] = z

        # randomize_heading(sampled_cube_state)
        sampled_cube_state[:, 3] = 0.0
        sampled_cube_state[:, 4] = 0.0
        sampled_cube_state[:, 5] = 0.0

        sampled_cube_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
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
                cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                
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
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                        2.0 * self.start_position_noise * (
                                                torch.rand(num_resets, 2, device=self.device) - 0.5)
            # sampled_cube_state[:, :2] = centered_cube_xy_state[:2] #+ 0.2

        this_cube_state_all[env_ids, :] = sampled_cube_state

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
        # print(actions)
        # step physics and render each frame
        mean_torques = torch.zeros((self.torques.shape[0], self.torques.shape[1]), device=self.torques.device)
        for i in range(self.decimation):
            torques = self.compute_torques(self._pos_control)
            # if i == 0:
            #     self.torques = torques.clone()
            upper_torque_limit = torch.tensor([10, 15, 15, 10, 10, 10, 10]).to(self.device)
            torques = torch.clip(torques, min=-upper_torque_limit, max=upper_torque_limit)
            self.torques = torques.clone()
            mean_torques += torques.abs()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques.detach()))
            self.states['previous_q'] = self.states['q'].clone()

            self.gym.simulate(self.sim)
            # self.gym.fetch_results(self.sim, True)
            # self.gym.step_graphics(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self._update_states()
            # if self.force_render:
            self.render()

        # mean_torques /= self.decimation
        # self.torques = torques
        # print(mean_torques.mean(dim=0))

        # to fix!
        # if self.device == 'cpu':
        self.gym.fetch_results(self.sim, True)
            # self.gym.step_graphics(self.sim)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.extras['depth_images'] = self.depth_images.to(self.rl_device)
        self.extras['_cubeA_state'] =  self._cubeA_state[:, :7].to(self.rl_device)


        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def compute_torques(self, actions):
        # kp = torch.tensor([50., 50., 50., 50., 50., 50., 20.]).cuda()
        # kd = torch.tensor([5., 5., 5., 5., 5., 5., 1.]).cuda()
        kp = torch.tensor([30., 30., 30., 30., 30., 30., 20.]).cuda()
        kd = torch.tensor([1., 1., 1., 1., 1., 1., 1.]).cuda()
        torques = kp * (actions - self.states['q']) - kd * self.states['q_vel']
        return torques

    def pre_physics_step(self, actions):
        # self.actions = actions.clone().to(self.device)
        # u_arm = self.actions

        # scale = 0.5 * (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
        # offset = 0.5 * (self.franka_dof_lower_limits + self.franka_dof_upper_limits)

        # u_arm = offset + scale * actions

        self.actions = self.actions.clone() + (1/200) * actions * 7.5
        self.actions = self.actions.clip(self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        self._arm_control[:, :] = self.actions

        #self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):
        self.progress_buf += 1
        self.running_epoch += 1
        self.current_episode = torch.tensor(self.running_epoch // 32)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.previous_actions = self.actions.clone()
        self.previous_cubeA_pos = self.states["cubeA_pos"].clone()
        self.previous_eef_pos = self.states["eef_pos"].clone()

        if not self.test_mode and self.current_episode != 0 and self.current_episode % 20 == 0 and (
                self.use_pointcloud or self.use_depth_sensors) and self.use_autoencoder and self.train_autoencoder:
            torch.save(self.autoencoder.state_dict(),
                       f'/home/guy/VSprojects/pointcloud_dexterity/isaacgymenvs/autoencoder_checkpoints/{self.autoencoder.input_type}_autoencoder_z_{self.z_size}_episode_{self.current_episode}.pkl')

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                                       [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                                       [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                                       [0.1, 0.1, 0.85])


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


#@torch.jit.script
def compute_franka_reward(
        reset_buf, progress_buf, actions, states, reward_settings, max_episode_length, current_episode, segmentation_sensors
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
    cubeA_size = states["cubeA_size"]
    cubeB_size = states["cubeB_size"]
    test_mode = states["cubeB_size"].shape[0] < 100
    is_cube_in_frame = ((segmentation_sensors.view(segmentation_sensors.shape[0], -1) == 2).sum(-1) > 0).int()

    # result = my_quat_rotate(states['eef_quat'], torch.tensor([0., 0., 1.], device=states['eef_quat'].device).repeat(
    #     states['eef_quat'].shape[0], 1))
    # result_b = my_quat_rotate(states['eef_quat'], torch.tensor([1., 0., 0.], device=states['eef_quat'].device).repeat(
    #     states['eef_quat'].shape[0], 1))

    # from_side = torch.sum(result * torch.tensor([0., 0., 1.], device=result.device), dim=-1).abs()
    # from_side_b = torch.sum(result_b * torch.tensor([0., 0., 1.], device=result.device), dim=-1).abs()

    not_laying_x_ = my_quat_rotate(states['cubeA_quat'],
                                   torch.tensor([0., 1., 0.], device=states['cubeA_quat'].device).repeat(
                                       states['cubeA_quat'].shape[0], 1))
    not_laying_y_ = my_quat_rotate(states['cubeA_quat'],
                                   torch.tensor([1., 0., 0.], device=states['cubeA_quat'].device).repeat(
                                       states['cubeA_quat'].shape[0], 1))

    not_laying_x = torch.sum(not_laying_x_ * torch.tensor([0., 0., 1.], device=cubeA_size.device), dim=-1).abs()
    not_laying_y = torch.sum(not_laying_y_ * torch.tensor([0., 0., 1.], device=cubeA_size.device), dim=-1).abs()

    d = torch.norm(states["cubeA_pos"] - states["eef_pos"], dim=-1)
    distance_of_cube_from_init = torch.norm(states["cubeA_pos"][:, :2] - states["cubeA_init_pos"][:, :2], dim=-1)

    norm_torq = torch.nn.functional.normalize(states['mean_torque'][:, :4], dim=0)
    norm_eef_vel = torch.nn.functional.normalize(states['eef_vel'][:, :3].abs(), dim=0)
    # print(d)
    dist_reward = 1 - torch.tanh(4. * (d))
    close_dist_flag = d < 0.25

    # print(d.mean())
    table_h = reward_settings["table_height"]
    cubeA_height = states["cubeA_pos"][:, 2] - table_h

    # stay_at_place = states["cubeA_pos"][:, 0]

    # torque_reward = torch.exp(-0.05 * torch.norm(states['mean_torque'], dim=-1))
    # action_diff = torch.exp(-torch.norm(actions - states['previous_action'], dim=-1))
    # cubaA_diff = torch.exp(-torch.norm(states["cubeA_pos"][:, :2] - states['previous_cubeA_pos'][:, :2], dim=-1))

    # print(cubeA_height - cubeA_size)

    # print(lift_reward)
    # print(dist_reward.mean().item())#, (cubeA_height - cubeA_size).float().mean().item())
    # how closely aligned cubeA is to cubeB (only provided if cubeA is lifted)
    # offset = torch.zeros_like(states["cubeA_pos"])
    # offset[:, 2] = (cubeA_size + cubeB_size) / 2
    # d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
    # align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted

    # dist_reward = torch.max(dist_reward, align_reward)

    # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
    # cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.05)
    # cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
    # gripper_away_from_cubeA = (d > 0.05)
    # stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA
    cubeA_lifted = (cubeA_height - cubeA_size) > 0.1

    # rewards = 3 * dist_reward + close_dist_flag*(40 * (cubeA_height - cubeA_size)*(cubeA_height<0.3) + 80 *(cubeA_height>=0.3)) + 5*action_diff  + torque_reward + cubaA_diff
    # rewards = 3 * dist_reward + close_dist_flag*40 * (cubeA_height - cubeA_size) + 5*action_diff  + torque_reward + 0.3*cubaA_diff
    # rewards = 3 * dist_reward + close_dist_flag*80 * (cubeA_height - cubeA_size)  + 0.2*norm_eef_vel.mean(1)/norm_torq.mean(1) + 5*action_diff #+ 10*action_diff  + cubaA_diff

    part1 = 0.1*(d < 0.25) #+ 1*dist_reward #+ 10*(d < 0.04) #+3 * is_cube_in_frame * dist_reward *(d < 0.05)#dist_reward
    part2 = 2. * close_dist_flag* (cubeA_height-0.01) #* (
                #cubeA_height < 0.3)  # close_dist_flag*60. * (cubeA_height)*(cubeA_height<0.3)
    part3 = close_dist_flag * 80. * (cubeA_height >= 0.3)
    part4 = 2*cubeA_lifted
    part5 =  2* is_cube_in_frame #*dist_reward #* (d >= 0.04) #states['mean_torque']  # .mean(1) #- 0.05*
    part6 = 5* (segmentation_sensors.view(segmentation_sensors.shape[0], -1) == 2).sum(-1) / (segmentation_sensors.shape[1]*segmentation_sensors.shape[1])#* (-not_laying_x - not_laying_y)
    part7 = -0.5 * distance_of_cube_from_init #+ 3*torch.exp(-not_laying_x - not_laying_y)
    part8 = -1*((torch.exp(-not_laying_x ) < 0.85) |  (torch.exp(-not_laying_y ) < 0.85)) #*(d >= 0.04)  # *(cubeA_height>=0.06)
    #print(part1, part2, part3)
    #rewards = torch.sum(torch.cat((part1, part2, part3), -1),-1) # , part2, part3, part4, part5
    rewards = sum([part1, part2, part4]) #part1, part2, part3, part5, 
    # rewards =3 * dist_reward +  close_dist_flag*(80 * (cubeA_height - cubeA_size)*(cubeA_height<0.3) + 80*(cubeA_height>=0.3)) + 10*action_diff - 0.005*states['mean_torque'].mean(1)  #-from_side -from_side_b #+ cubaA_diff + 1*torque_reward #gpu0 - norm_torq.mean(1)
    # rewards = - states["q"][:,-1]  + 0.1* dist_reward
    # print(states['mean_torque'][-1])
    # print((actions - states['previous_action'])[-1])
    # print(200*(states['eef_pos'] - states['previous_eef_pos']).abs()[-1])
    # print(states['eef_vel'][-1,:3].abs())
    # print(states['mean_torque'][-1])
    # print('--------')
    # print(states['eef_vel'][-1])
    # print(states['mean_torque'].mean(0))
    
    if test_mode:
        print(d[-10:])
        # print('cylinder:')
        # print(states["cubeA_pos"][-1])
        # print(states["to_target"][-1])
        # #print(d[-1])
        # for idx, part_r in enumerate([part1, part2, part3, part4, part5, part6, part7, part8]):
        #     print(idx+1, part_r[-1])

        # print('cylinder2:')
        # print(states["cubeA_pos"][-3])
        # print(states["to_target"][-3])
        # #print(d[-3])
        # for idx, part_r in enumerate([part1, part2, part3, part4, part5, part6, part7, part8]):
        #     print(idx+1, part_r[-3])

        print('plate:')
        print(states["cubeA_pos"][-2])
        print(states["to_target"][-2])
        #print(d[-4])
        for idx, part_r in enumerate([part1, part2, part3, part4, part5, part6, part7, part8]):
            print(idx+1, part_r[-2])
        print('-------')
    # print(0.005*states['mean_torque'].mean(dim=0).mean())
    # print(states['mean_torque'].max(dim=0)[0])
    # print(states['mean_torque'].min(dim=0)[0])
    # print(states['eef_vel'].mean(dim=0))
    # print(states['eef_vel'].max(dim=0)[0])
    # print(states['eef_vel'].min(dim=0)[0])
    # print('****')
    # print(norm_torq.mean(0))
    # print(norm_eef_vel.mean(0))
    # print((norm_eef_vel.mean(1)/norm_torq.mean(1))[-1] )
    # print('------------')
    # print(states['cubeA_vel'][-1, :3].mean())
    # print(states['eef_vel'][-1])#, :3].abs().mean(), states['eef_vel'][-1, 3:7].abs().mean())
    # print(states['mean_torque'][-1].mean())
    # print(norm_eef_vel[-1].mean()/norm_torq[-1].mean())

    # print('===========')

    # aaa = [(a.item(),b.item(),c.item()) for a,b,c in zip(states["q"].mean(0),states["q"].max(0)[0], states["q"].min(0)[0] )]
    # for aa in aaa:
    #     print(aa)
    # print('-------')
    # rewards = torch.where(
    #     stack_reward ,
    #     reward_settings["r_stack_scale"] * stack_reward  ,
    #     reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings[
    #         "r_align_scale"] * align_reward,
    # )

    # Compute resets
    reset_rigid_body_h = table_h + 0.0326 + 0.025  # height of the joints radius
    # print(reset_rigid_body_h)
    # print(states['link03_pos'][-1, 2].item(), states['link04_pos'][-1, 2].item(), states['link05_pos'][-1, 2].item(),states['link06_pos'][-1, 2].item())

    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
    cube_fall = states['cubeA_pos'][:, 2] < 1
    reset_buf |= cube_fall
    reset_buf |= (cubeA_height - cubeA_size) > 0.5
    # reset_buf |= ((states['link03_pos'][:, 2] < reset_rigid_body_h) | (states['link04_pos'][:, 2] < reset_rigid_body_h) | (states['link05_pos'][:, 2] < table_h + 0.038) | (states['link06_pos'][:, 2] < reset_rigid_body_h))# | (states['eef_pos'][:, 2] < reset_rigid_body_h) )
    # reset_buf |= ((states['link03_pos'][:, 2] < reset_rigid_body_h) | (states['link04_pos'][:, 2] < reset_rigid_body_h) | (states['link05_pos'][:, 2] < reset_rigid_body_h) | (states['link06_pos'][:, 2] < reset_rigid_body_h) | (states['eef_pos'][:, 2] < reset_rigid_body_h) ) #05: table_h + 0.038
    # reset_buf |= norm_eef_vel.mean(1)/norm_torq.mean(1) > 1.1
    # reset_buf |= ((states['mean_torque'].max(dim=1)[0] > 60 ) & (states['eef_vel'].abs()[:,:3].mean(dim=1) < 0.005) )
    # reset_buf |= (states['cubeA_pos'][:, 0] < -0.40)
    # reset_buf |= (states['cubeA_pos'][:, 1] < -0.6)
    # reset_buf |= (states['cubeA_pos'][:, 1] > 0.6)
    return rewards, reset_buf


