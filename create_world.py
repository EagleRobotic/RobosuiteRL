from robosuite.robosuite.models import MujocoWorldBase
from robosuite.robosuite.models.robots import Panda
from robosuite.robosuite.models.grippers import gripper_factory
from robosuite.robosuite.models.arenas import TableArena
from robosuite.robosuite.models.objects import BallObject
from robosuite.robosuite.utils.mjcf_utils import new_joint
from mujoco_py import MjSim, MjViewer
from robosuite.robosuite.controllers import osc, load_controller_config
import numpy as np

import robosuite.robosuite as suite
from robosuite.robosuite.controllers import load_controller_config
from robosuite.robosuite.utils.input_utils import input2action
from robosuite.robosuite.wrappers import VisualizationWrapper
from robosuite.robosuite.devices import Keyboard

#Configuration of the world
# world = MujocoWorldBase()
# mujoco_robot = Panda()
# gripper = gripper_factory('PandaGripper')
# mujoco_robot.add_gripper(gripper)

# mujoco_robot.set_base_xpos([0.5, 0.3, 0.7])
# world.merge(mujoco_robot)

# mujoco_arena = TableArena()
# mujoco_arena.set_origin([0.8, 0, 0])
# world.merge(mujoco_arena)

# sphere = BallObject(
#     name="sphere",
#     size=[0.04],
#     rgba=[0, 0.5, 0.5, 1]).get_obj()
# sphere.set('pos', '1.0 0 1.0')
# world.worldbody.append(sphere)

# #Getting configured world
# model = world.get_model(mode="mujoco_py")

# sim = MjSim(model)
# viewer = MjViewer(sim)
# viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

# for i in range(10000):
#   sim.data.ctrl[:] = 0
#   sim.step()
#   viewer.render()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--environment", type=str, default="Lift")
parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
parser.add_argument(
    "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
)
parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
parser.add_argument("--device", type=str, default="keyboard")
parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
args = parser.parse_args()


controller_name = "OSC_POSE"
controller_config = load_controller_config(default_controller=controller_name)

# Create argument configuration
config = {
    "env_name": "Door",
    "robots": "Panda",
    "controller_configs": controller_config,
}

# Create environment
env = suite.make(
    **config,
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="agentview",
    ignore_done=True,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
    hard_reset=False,
)

# Wrap this environment in a visualization wrapper
env = VisualizationWrapper(env, indicator_configs=None)

# Setup printing options for numbers
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
env.viewer.add_keypress_callback("any", device.on_press)
env.viewer.add_keyup_callback("any", device.on_release)
env.viewer.add_keyrepeat_callback("any", device.on_press)


while True:
    # Reset the environment
    obs = env.reset()

    # Setup rendering
    cam_id = 0
    num_cam = len(env.sim.model.camera_names)
    env.render()

    # Initialize variables that should the maintained between resets
    last_grasp = 0

    # Initialize device control
    device.start_control()

    while True:
        # Set active robot
        active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config
        )

        # If action is none, then this a reset so we should break
        if action is None:
            break

        # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
        # toggle arm control and / or camera viewing angle if requested
        if last_grasp < 0 < grasp:
            if args.switch_on_grasp:
                args.arm = "left" if args.arm == "right" else "right"
            if args.toggle_camera_on_grasp:
                cam_id = (cam_id + 1) % num_cam
                env.viewer.set_camera(camera_id=cam_id)
        # Update last grasp
        last_grasp = grasp

        # Fill out the rest of the action space if necessary
        rem_action_dim = env.action_dim - action.size
        if rem_action_dim > 0:
            # Initialize remaining action space
            rem_action = np.zeros(rem_action_dim)
            # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
            if args.arm == "right":
                action = np.concatenate([action, rem_action])
            elif args.arm == "left":
                action = np.concatenate([rem_action, action])
            else:
                # Only right and left arms supported
                print(
                    "Error: Unsupported arm specified -- "
                    "must be either 'right' or 'left'! Got: {}".format(args.arm)
                )
        elif rem_action_dim < 0:
            # We're in an environment with no gripper action space, so trim the action space to be the action dim
            action = action[: env.action_dim]

        # Step through the simulation and render
        obs, reward, done, info = env.step(action)
        env.render()
