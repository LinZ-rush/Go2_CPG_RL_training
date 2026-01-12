from legged_gym.envs.base.jump_config import JumpCfg, JumpCfgPPO

class GO2JumpCfg( JumpCfg ):
    class init_state( JumpCfg.init_state ):
        pos = [0.0, 0.0, 0.32] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( JumpCfg.control ):
        # PD Drive parameters:
        control_type = 'CPG_OFFSETX'
        stiffness = {'joint': 100.}  # [N*m/rad]
        damping = {'joint': 2}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset( JumpCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        hip_link_length_go2 = 0.0955
        thigh_link_length_go2 = 0.213
        calf_link_length_go2 = 0.213


class GO2JumpCfgPPO( JumpCfgPPO ):
    class algorithm( JumpCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( JumpCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2_high_level'

  
