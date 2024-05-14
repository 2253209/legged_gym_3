# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Zq12Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 47  # 169
        num_actions = 12
        env_spacing = 1.
        queue_len_obs = 4
        queue_len_act = 4
        obs_latency = [5, 20]
        act_latency = [5, 20]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.84]  # x,y,z [m] accurate:0.832
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'JOINT_Y1': -0.0,
            'JOINT_Y2': 0.0,
            'JOINT_Y3': 0.21,
            'JOINT_Y4': -0.53,
            'JOINT_Y5': 0.31,
            'JOINT_Y6': 0.0,

            'JOINT_Z1': 0.0,
            'JOINT_Z2': 0.0,
            'JOINT_Z3': 0.21,
            'JOINT_Z4': -0.53,
            'JOINT_Z5': 0.31,
            'JOINT_Z6': -0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # stiffness = {'JOINT': 100.0}  # [N*m/rad]
        # damping = {'JOINT': 0.0}
        stiffness = {'JOINT_Y1': 160.0, 'JOINT_Y2': 160.0, 'JOINT_Y3': 160.0, 'JOINT_Y4': 160.0, 'JOINT_Y5': 18.0, 'JOINT_Y6': 18.0,
                     'JOINT_Z1': 160.0, 'JOINT_Z2': 160.0, 'JOINT_Z3': 160.0, 'JOINT_Z4': 160.0, 'JOINT_Z5': 18.0, 'JOINT_Z6': 18.0,
                     }  # [N*m/rad]
        damping = {'JOINT_Y1': 10.0, 'JOINT_Y2': 10.0, 'JOINT_Y3': 10.0, 'JOINT_Y4': 10.0, 'JOINT_Y5': 4.0, 'JOINT_Y6': 4.0,
                   'JOINT_Z1': 10.0, 'JOINT_Z2': 10.0, 'JOINT_Z3': 10.0, 'JOINT_Z4': 10.0, 'JOINT_Z5': 4.0, 'JOINT_Z6': 4.0,
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

        actuator_inertia = 0.042
        ankle_actuator_inertia = 0.035

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        # gravity = [0., 0., 0]  # [m/s^2]
        class physx(LeggedRobotCfg.sim.physx):
            contact_offset = 0.001  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.05  # 0.5 [m/s]
            friction_offset_threshold = 0.002


    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [-3, -3, 3]  # [m]
        lookat = [0., 0, 1.]  # [m]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.25
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 50.
        clip_actions = 100.

    class commands(LeggedRobotCfg.commands):
        step_joint_offset = 0.50  # rad
        step_freq = 0.2  # HZ （e.g. cycle-time=0.66）

        class ranges(LeggedRobotCfg.commands.ranges):
            #lin_vel_x = [-0.3, 0.5]  # min max [m/s]
            #lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            #ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            #heading = [-3.14, 3.14]
            lin_vel_x = [-0.0, 0.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]  # min max [rad/s]
            heading = [-0, 0]
        heading_command = False  # if true: compute ang vel command from heading error

    class asset(LeggedRobotCfg.asset):
        # file = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/zq01/mjcf/zq_box_foot.xml'
        file = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/ZQ_Humanoid/urdf/ZQ_Humanoid_long_foot.urdf'
        name = "zq01"
        foot_name = 'foot'
        penalize_contacts_on = ['3', '4']
        terminate_after_contacts_on = []
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        terminate_body_height = 0.4
        disable_gravity = False
        fix_base_link = False
        armature = 0.0001
        thickness = 0.001

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [1.0, 1.5]

        randomize_body_mass = True
        added_body_mass_range = [-0.20, 0.75]
        added_leg_mass_range = [-0.1, 0.30]

        randomize_body_com = True
        added_body_com_range = [-0.02, 0.02]  # 1cm
        added_leg_com_range = [-0.01, 0.01]  # 0.5cm

        randomize_body_inertia = True
        scaled_body_inertia_range = [0.90, 1.1]  # %5 error

        randomize_motor_strength = True
        scaled_motor_strength_range = [0.80, 1.2]  # %5 error

        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.4
        max_push_vel_ang = 0.10
        push_curriculum_start_step = 10000*24
        push_curriculum_common_step = 30000*24

        randomize_init_state = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.80
        soft_torque_limit = 0.80
        max_contact_force = 350.
        only_positive_rewards = True
        base_height_target = 0.83
        tracking_sigma = 0.15
        min_dist = 0.2
        max_dist = 0.5

        class scales(LeggedRobotCfg.rewards.scales):

            termination = -5.  # 4. 不倒
            tracking_lin_vel = 0.
            tracking_lin_x_vel = 5.0  # 6. 奖励速度为0
            tracking_lin_y_vel = 4.0  # 6. 奖励速度为0
            tracking_ang_vel = 5.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = -5.0  # 5. 重力投影
            #
            action_smoothness = -0.  # 0.002
            torques = -3.0e-5
            dof_vel = -0.0
            dof_acc = -2.5e-6
            #
            base_height = -0.0  # 1.奖励高度？惩罚高度方差
            feet_air_time = 0.
            collision = -20
            dof_pos_limits = -100.  # 让各个关节不要到达最大位置
            torque_limits = -2.0
            #
            feet_stumble = -0.0
            feet_contact_forces = -0.01
            #
            action_rate = -0.25
            stand_still = -0.  # 3. 惩罚：0指令运动。关节角度偏离 初始值
            no_fly = 0.0  # 2. 奖励：两脚都在地上，有一定压力
            target_joint_pos = 10.0  # 3.  reference joint imitation reward
            # body_feet_dist = -1.0
            feet_distance = 1.0

            # penalty the ankle shaking
            ankle_dof_acc = -0.15e-3
            ankle_action_rate = -0.15

            target_hip_roll_pos = 2.0
            target_ankle_pos = 0.5





class Zq12CfgPPO(LeggedRobotCfgPPO):

    init_noise_std = 0.12
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'tanh'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'zq12sq'
        max_iterations = 10000
        # logging
        save_interval = 400
        # checkpoint = '90000'
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.e-4  # 5.e-4
        schedule = 'fixed'  # could be adaptive, fixed
