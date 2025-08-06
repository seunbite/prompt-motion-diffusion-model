import sys
import os
import numpy as np
from tqdm import trange
from typing import Dict, List, Tuple, Optional
import cv2

import mujoco
sys.path.append('../exercise-yet-another-mujoco-tutorial-v3/package/helper/')
sys.path.append('../exercise-yet-another-mujoco-tutorial-v3/package/mujoco_usage/')
sys.path.append('../exercise-yet-another-mujoco-tutorial-v3/package/motion_retarget/')
from mujoco_parser import MuJoCoParserClass, init_ik_info, add_ik_info, get_dq_from_ik_info
from utility import *
from transformation import *
from joi import *
from mr_cmu import *
from scipy.spatial.transform import Rotation as R
import imageio


def ensure_T_joi_src_is_4x4(T_joi_src):
    # 이미 (N, 4, 4)면 그대로 반환
    if T_joi_src.ndim == 3 and T_joi_src.shape[1:] == (4, 4):
        return T_joi_src
    # (N, 3) 좌표만 있는 경우 → 4x4 행렬로 변환 (회전은 단위행렬)
    elif T_joi_src.ndim == 2 and T_joi_src.shape[1] == 3:
        N = T_joi_src.shape[0]
        T = np.tile(np.eye(4), (N, 1, 1))
        T[:, :3, 3] = T_joi_src
        return T
    # (N, 3, 3) 회전행렬만 있는 경우 → 4x4로 변환
    elif T_joi_src.ndim == 3 and T_joi_src.shape[1:] == (3, 3):
        N = T_joi_src.shape[0]
        T = np.tile(np.eye(4), (N, 1, 1))
        T[:, :3, :3] = T_joi_src
        return T
    # (joints, features) 형태인 경우 → (joints, 4, 4)로 변환
    elif T_joi_src.ndim == 2:
        N_joints = T_joi_src.shape[0]
        N_features = T_joi_src.shape[1]
        
        # 각 joint에 대해 4x4 변환 행렬 생성
        T = np.tile(np.eye(4), (N_joints, 1, 1))
        
        # features를 4x4 행렬로 재구성
        if N_features >= 16:
            # 16개씩 묶어서 4x4 행렬로 변환
            for i in range(N_joints):
                if i * 16 < N_features:
                    features = T_joi_src[i, :16]
                    T[i] = features.reshape(4, 4)
        elif N_features >= 12:
            # 12개씩 묶어서 3x4 행렬로 변환 후 4x4로 확장
            for i in range(N_joints):
                if i * 12 < N_features:
                    features = T_joi_src[i, :12]
                    T_3x4 = features.reshape(3, 4)
                    T[i, :3, :] = T_3x4
        elif N_features >= 6:
            # 6개씩 묶어서 3x3 회전행렬로 변환 후 4x4로 확장
            for i in range(N_joints):
                if i * 6 < N_features:
                    features = T_joi_src[i, :6]
                    R_3x3 = features.reshape(3, 3)
                    T[i, :3, :3] = R_3x3
        else:
            # 단순히 위치 정보만 사용 (첫 3개 features를 위치로)
            for i in range(N_joints):
                if N_features >= 3:
                    T[i, :3, 3] = T_joi_src[i, :3]
        return T
    else:
        raise ValueError(f"Unknown T_joi_src shape: {T_joi_src.shape}")


class MDMToHumanoidRetargeter():
    """MDM 모션을 휴머노이드 로봇으로 리타게팅"""
    # super init
    def __init__(self, env: MuJoCoParserClass):
        self.env = env
        self.env.reset(step=True)
        self.smpl_to_g1_idx = {
            'hip': 0,      # pelvis
            'neck': 12,    # neck  
            'rs': 17,      # right_shoulder
            're': 19,      # right_elbow
            'rw': 21,      # right_wrist
            'ls': 16,      # left_shoulder
            'le': 18,      # left_elbow
            'lw': 20,      # left_wrist
            'rp': 2,       # right_hip
            'rk': 5,       # right_knee
            'ra': 8,       # right_ankle
            'lp': 1,       # left_hip
            'lk': 4,       # left_knee
            'la': 7        # left_ankle
        }
        self.width = 640
        self.height = 480
        self.camera_distance = 4


        self.free_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.free_cam)
        self.free_cam.azimuth = 90      # 좌우 회전 (원하는 값으로 조절)
        self.free_cam.elevation = -15   # 위/아래 각도
        self.free_cam.distance = self.camera_distance

    def auto_scale_and_rotation(self, motion_data: np.ndarray, env: MuJoCoParserClass):
        source_height, source_orientation, source_center = self._calculate_source_dimensions(motion_data, ensure=True)
        target_height, target_orientation, target_center = self._calculate_target_dimensions(env)
        
        position_offset = target_center - source_center
        scale = target_height / source_height if source_height > 0 else 1.0
        scale = np.clip(scale, 0.1, 10.0)
        
        rotation = self._calculate_rotation_difference(source_orientation, target_orientation)
        rotation = tuple(np.clip(rot, -180, 180) for rot in rotation)
        
        print(f"🔍 자동 위치/스케일/회전 계산:")
        print(f"   소스 중심: {source_center}, 키: {source_height:.3f}m, 방향: {source_orientation:.1f}°")
        print(f"   타겟 중심: {target_center}, 키: {target_height:.3f}m, 방향: {target_orientation:.1f}°")
        
        return position_offset, scale, rotation
            
    
    def _calculate_source_dimensions(self, motion_data: np.ndarray, ensure: bool = False) -> Tuple[float, float, np.ndarray]:
        head_pos = motion_data[0, self.smpl_to_g1_idx['neck'], :3]
        left_foot_pos = motion_data[0, self.smpl_to_g1_idx['la'], :3]
        right_foot_pos = motion_data[0, self.smpl_to_g1_idx['ra'], :3]
        foot_center = (left_foot_pos + right_foot_pos) / 2
        center = (head_pos + foot_center) / 2
        
        height = np.linalg.norm(head_pos - foot_center)
        
        head_foot_vector = head_pos - foot_center
        horizontal_angle = np.degrees(np.arctan2(head_foot_vector[1], head_foot_vector[0]))
        
        foot_direction = right_foot_pos - left_foot_pos
        foot_angle = np.degrees(np.arctan2(foot_direction[1], foot_direction[0]))
        
        overall_orientation = (horizontal_angle + foot_angle) / 2
        
        return height, overall_orientation, center
    
    def _calculate_target_dimensions(self, env: MuJoCoParserClass) -> Tuple[float, float, np.ndarray]:
        """
        타겟 로봇에서 키, 방향, 중심점 계산
        
        Returns:
            height: 머리에서 발까지의 거리 (미터)
            orientation: 머리-발 방향 각도 (도)
            center: 로봇의 중심점 (x, y, z)
        """
        # 로봇의 현재 관절 위치 가져오기
        T_joi_trgt = get_T_joi_from_g1(env)
        
        # 머리(neck)와 발(ankle) 위치 추출
        head_pos = t2p(T_joi_trgt['neck'])
        left_foot_pos = t2p(T_joi_trgt['la'])
        right_foot_pos = t2p(T_joi_trgt['ra'])
        
        # 발 중간점 계산
        foot_center = (left_foot_pos + right_foot_pos) / 2
        
        # 중심점 계산 (머리와 발의 중간점)
        center = (head_pos + foot_center) / 2
        
        # 키 계산 (머리에서 발까지의 거리)
        height = np.linalg.norm(head_pos - foot_center)
        
        # 방향 계산 (머리-발 벡터의 수평면에서의 각도)
        head_foot_vector = head_pos - foot_center
        horizontal_angle = np.degrees(np.arctan2(head_foot_vector[1], head_foot_vector[0]))
        
        # 발꿈치-발앞꿈치 방향도 고려 (발의 방향)
        foot_direction = right_foot_pos - left_foot_pos
        foot_angle = np.degrees(np.arctan2(foot_direction[1], foot_direction[0]))
        
        # 전체 방향은 머리-발 방향과 발 방향의 평균
        overall_orientation = (horizontal_angle + foot_angle) / 2
        
        return height, overall_orientation, center
    
    def _calculate_rotation_difference(self, source_orientation: float, target_orientation: float) -> Tuple[float, float, float]:
        """
        소스와 타겟 방향의 차이를 계산하여 회전 각도 반환
        
        Args:
            source_orientation: 소스 방향 각도 (도)
            target_orientation: 타겟 방향 각도 (도)
            
        Returns:
            rotation: (x, y, z) 회전 각도 (도)
        """
        # Y축 회전 (수평면에서의 회전)
        y_rotation = target_orientation - source_orientation
        
        # 각도를 -180 ~ 180 범위로 정규화
        while y_rotation > 180:
            y_rotation -= 360
        while y_rotation < -180:
            y_rotation += 360
        
        # X축과 Z축 회전은 0으로 설정 (필요시 추가 계산 가능)
        x_rotation = 0.0
        z_rotation = 0.0
        
        return (x_rotation, y_rotation, z_rotation)

    def apply_scale_and_rotation(self, motion_data: np.ndarray, position_offset: np.ndarray = None, scale: float = None, rotation: Tuple[float, float, float] = None):
        if position_offset is None or scale is None or rotation is None:
            position_offset, scale, rotation = self.auto_scale_and_rotation(motion_data, self.env)
        
        rot_matrix = R.from_euler('xyz', rotation, degrees=True).as_matrix()
        
        if motion_data.ndim == 4:
            # (batch, joints, features, frames)
            for i in range(motion_data.shape[0]):
                for j in range(motion_data.shape[1]):
                    for k in range(motion_data.shape[3]):
                        # 위치 정보 추출 (처음 3개 요소가 위치)
                        pos = motion_data[i, j, :3, k]
                        
                        # 변환 적용: 위치 오프셋 + 스케일 + 회전
                        pos = (pos * scale) + position_offset
                        pos = rot_matrix @ pos
                        
                        # 변환된 위치를 다시 저장
                        motion_data[i, j, :3, k] = pos
                        
        elif motion_data.ndim == 3: # (frames, joints, features)
            for i in range(motion_data.shape[0]):
                for j in range(motion_data.shape[1]):
                    # 위치 정보 추출 (처음 3개 요소가 위치)
                    pos = motion_data[i, j, :3]
                    
                    # 변환 적용: 위치 오프셋 + 스케일 + 회전
                    pos = (pos * scale) + position_offset
                    pos = rot_matrix @ pos
                    
                    # 변환된 위치를 다시 저장
                    motion_data[i, j, :3] = pos
        else:
            raise ValueError(f"Unexpected motion data shape: {motion_data.shape}")
        
        return motion_data

    def retarget_g1_mujoco(
        self, 
        motion_data: np.ndarray, 
        update_base_every_frame: bool = False, 
        visualize: bool = False, 
        filename: str = 'save/retargeted_motion.mp4', 
        fps: int = 20,
        show_target_spheres: bool = True,
        position_offset: Optional[np.ndarray] = None,
        scale: Optional[float] = None,
        rotation: Optional[Tuple[float, float, float]] = None
        ):
        if self.env is None:
            raise ValueError("MuJoCo 환경이 없습니다.")
        
        motion_data = self.apply_scale_and_rotation(motion_data, position_offset, scale, rotation)
        
        qpos_list = []
        video_frames = []
        renderer = mujoco.Renderer(self.env.model, width=self.width, height=self.height)
        frames = motion_data.shape[0]
        target_spheres_list = []
        if visualize:
            self.env.init_viewer(title='MDM to G1 Motion Retargeting', transparent=True)
        
        tick = 0
        for tick in trange(frames):
            T_joi_src = motion_data[tick, :, :] 
            T_joi_src = ensure_T_joi_src_is_4x4(T_joi_src)
            T_joi_src = {k: T_joi_src[v, :, :] for k, v in self.smpl_to_g1_idx.items()}

            if update_base_every_frame or tick == 0:
                T_base_src  = T_joi_src['hip']
                T_base_trgt = T_yuzf2zuxf(T_base_src)
                
                self.env.set_T_base_body(body_name='pelvis',T=T_base_trgt)
                self.env.forward()
                # self.env.step()

            # Retargeting
            T_joi_trgt = get_T_joi_from_g1(self.env)
            len_hip2neck   = len_T_joi(T_joi_trgt,'hip','neck')
            len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
            len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
            len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
            len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
            len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
            len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
            len_hip2rp     = len_T_joi(T_joi_trgt,'hip','rp')
            len_rp2rk      = len_T_joi(T_joi_trgt,'rp','rk')
            len_rk2ra      = len_T_joi(T_joi_trgt,'rk','ra')
            len_hip2lp     = len_T_joi(T_joi_trgt,'hip','lp')
            len_lp2lk      = len_T_joi(T_joi_trgt,'lp','lk')
            len_lk2la      = len_T_joi(T_joi_trgt,'lk','la')
            
            rev_joint_names_for_ik_full_body = self.env.rev_joint_names
            joint_idxs_jac_full_body = self.env.get_idxs_jac(
                joint_names=rev_joint_names_for_ik_full_body)
            joint_idxs_jac_full_body_with_base = np.concatenate(
                ([0,1,2,3,4,5],joint_idxs_jac_full_body))

            uv_hip2neck   = uv_T_joi(T_joi_src,'hip','neck')
            uv_neck2rs    = uv_T_joi(T_joi_src,'neck','rs')
            uv_rs2re      = uv_T_joi(T_joi_src,'rs','re')
            uv_re2rw      = uv_T_joi(T_joi_src,'re','rw')
            uv_neck2ls    = uv_T_joi(T_joi_src,'neck','ls')
            uv_ls2le      = uv_T_joi(T_joi_src,'ls','le')
            uv_le2lw      = uv_T_joi(T_joi_src,'le','lw')
            uv_hip2rp     = uv_T_joi(T_joi_src,'hip','rp')
            uv_rp2rk      = uv_T_joi(T_joi_src,'rp','rk')
            uv_rk2ra      = uv_T_joi(T_joi_src,'rk','ra')
            uv_hip2lp     = uv_T_joi(T_joi_src,'hip','lp')
            uv_lp2lk      = uv_T_joi(T_joi_src,'lp','lk')
            uv_lk2la      = uv_T_joi(T_joi_src,'lk','la')
            
            # Set positional targets with robust calculation
            try:
                if update_base_every_frame:
                    T_joi_trgt_current = get_T_joi_from_g1(self.env)
                    p_hip_current = t2p(T_joi_trgt_current['hip'])
                    p_hip_trgt = p_hip_current
                else:
                    p_hip_trgt = t2p(T_joi_src['hip'])
                
                p_neck_trgt  = p_hip_trgt + len_hip2neck*uv_hip2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                p_rp_trgt    = p_hip_trgt + len_hip2rp*uv_hip2rp
                p_rk_trgt    = p_rp_trgt + len_rp2rk*uv_rp2rk
                p_ra_trgt    = p_rk_trgt + len_rk2ra*uv_rk2ra
                p_lp_trgt    = p_hip_trgt + len_hip2lp*uv_hip2lp
                p_lk_trgt    = p_lp_trgt + len_lp2lk*uv_lp2lk
                p_la_trgt    = p_lk_trgt + len_lk2la*uv_lk2la
                
                target_positions = [p_neck_trgt, p_rs_trgt, p_re_trgt, p_rw_trgt, 
                                    p_ls_trgt, p_le_trgt, p_lw_trgt, p_rp_trgt, 
                                    p_rk_trgt, p_ra_trgt, p_lp_trgt, p_lk_trgt, p_la_trgt]
                target_spheres_list = {
                    'hip': p_hip_trgt,
                    'neck': p_neck_trgt,
                    'rs': p_rs_trgt,
                    're': p_re_trgt,
                    'rw': p_rw_trgt,
                    'ls': p_ls_trgt,
                    'le': p_le_trgt,
                    'lw': p_lw_trgt,
                    'rp': p_rp_trgt,
                    'rk': p_rk_trgt,
                    'ra': p_ra_trgt,
                    'lp': p_lp_trgt,
                    'lk': p_lk_trgt,
                    'la': p_la_trgt
                }
                
                invalid_positions = False
                for i, pos in enumerate(target_positions):
                    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                        print(f"Warning: Invalid target position {i} at tick {tick}, skipping frame")
                        invalid_positions = True
                        break
                
                if invalid_positions:
                    continue
                        
            except Exception as e:
                print(f"Error calculating target positions at tick {tick}: {e}")
                continue

            joi_body_name = get_joi_body_name_of_g1()
            ik_info_full_body = init_ik_info()
            add_ik_info(ik_info_full_body,body_name=joi_body_name['rs'],p_trgt=p_rs_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['re'],p_trgt=p_re_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['rw'],p_trgt=p_rw_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['ls'],p_trgt=p_ls_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['le'],p_trgt=p_le_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['lw'],p_trgt=p_lw_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['rp'],p_trgt=p_rp_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['rk'],p_trgt=p_rk_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['ra'],p_trgt=p_ra_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['lp'],p_trgt=p_lp_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['lk'],p_trgt=p_lk_trgt)
            add_ik_info(ik_info_full_body,body_name=joi_body_name['la'],p_trgt=p_la_trgt)

            max_ik_tick = 50
            ik_converged = False
            
            for ik_tick in range(max_ik_tick):
                dq,ik_err_stack = get_dq_from_ik_info(
                    env            = self.env,
                    ik_info        = ik_info_full_body,
                    stepsize       = 0.8,
                    eps            = 1e-2,
                    th             = np.radians(10.0),
                    joint_idxs_jac = joint_idxs_jac_full_body_with_base,
                )
                
                if np.any(np.isnan(dq)) or np.any(np.isinf(dq)):
                    print(f"Warning: Invalid dq at tick {tick}, ik_tick {ik_tick}")
                    break
                    
                qpos = self.env.get_qpos()
                qpos_backup = qpos.copy()
                
                mujoco.mj_integratePos(self.env.model, qpos, dq, 1)
                
                if np.any(np.isnan(qpos)) or np.any(np.isinf(qpos)):
                    print(f"Warning: Invalid qpos after integration at tick {tick}")
                    qpos = qpos_backup
                    break
                    
                self.env.forward(q=qpos)
                # self.env.step(ctrl=qpos, joint_names=self.env.joint_names)
                        
                # except Exception as e:
                #     print(f"Error in mj_integratePos at tick {tick}: {e}")
                #     qpos = qpos_backup
                #     break
                
                if np.linalg.norm(ik_err_stack) < 0.05:
                    ik_converged = True
                    break
                        
            status = "converged" if ik_converged else "failed"
            print(f"Tick {tick}: IK {status} after {ik_tick+1}/{max_ik_tick} iterations, Error: {np.linalg.norm(ik_err_stack):.4f}")
            qpos_list.append(self.env.get_qpos())
            video_frame = self.grab_image(target_spheres_list, renderer, show_target_spheres=show_target_spheres)
            video_frames.append(video_frame)
            
        if visualize:
            self.env.close_viewer()
            
        imageio.mimsave(filename, video_frames, fps=fps)
        print(f"[MuJoCo] mp4 저장 완료 (frames: {len(video_frames)}): {filename}")
        print ("Done.")
    
    def grab_image(self, target_spheres: Dict[str, np.ndarray], renderer: mujoco.Renderer, show_target_spheres: bool = True) -> np.ndarray:
        mujoco.mj_forward(self.env.model, self.env.data)

        # pelvis 위치를 중심으로 카메라 위치 업데이트
        pelvis_id = self.env.model.body('pelvis').id
        pelvis_pos = self.env.data.xpos[pelvis_id]
        self.free_cam.lookat[:] = pelvis_pos
        self.free_cam.distance = self.camera_distance  # 필요하면 매 프레임 다른 값도 가능

        # free camera 사용
        renderer.update_scene(self.env.data, camera=self.free_cam)
        frame = renderer.render()
        color_map = {
            k: (128, 128, 128) for k in target_spheres.keys() # gray
        }
        color_map['la'] = (0, 0, 0) # black
        color_map['ra'] = (0, 255, 0) # green
        color_map['hip'] = (255, 0, 0) # red
        color_map['neck'] = (0, 0, 255) # blue

        if show_target_spheres:
            cx, cy = self.width // 2, self.height // 2
            scale = 80.0
            for k, p in target_spheres.items():
                u = int(cx + p[0] * scale)
                v = int(cy - p[1] * scale)
                frame = cv2.circle(frame, (u, v), 5, color_map[k], -1)

        return frame


def retarget_motion(
    # motion_file = './save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_a_person_walks_forward/results.npy',
    motion_file = '/scratch2/iyy1112/motion-persona/save/20250805_mdm_type3/gpu_4/motion_1850_rep_00/motion.npy',
    output_file = './save/retargeted_motion.mp4',
    # output_file = '/scratch2/iyy1112/motion-persona/save/20250805_mdm_type3/gpu_4/motion_1849_rep_00/retarget_motion_1849_rep_00.mp4',
    xml_path = '../exercise-yet-another-mujoco-tutorial-v3/asset/unitree_g1/scene_g1.xml',
    # xml_path = '../exercise-yet-another-mujoco-tutorial-v3/asset/unitree_g1/g1.xml',
    position_offset = None,
    scale = None,
    rotation = None, # (89.7, 0, 0)
    show_target_spheres = True
):
    if not os.path.exists(motion_file):
        raise FileNotFoundError(f"💡 모션 파일을 찾을 수 없습니다: {motion_file}")
    else:
        data = np.load(motion_file, allow_pickle=True).item()
        
        # Handle both individual motion files and combined results files
        if 'motion' in data and isinstance(data['motion'], dict):
            # Individual motion file (from generate.py)
            motion_data = data['motion']['motion']
            print(f"🎯 개별 모션 파일에서 로드: {motion_data.shape}")
        elif 'motion' in data and isinstance(data['motion'], np.ndarray):
            # Combined results file
            motion_data = data['motion']
            print(f"🎯 통합 결과 파일에서 로드: {motion_data.shape}")
        else:
            # Direct motion data
            motion_data = data
            print(f"🎯 직접 모션 데이터: {motion_data.shape}")

    env = MuJoCoParserClass(name='G1_tmp', rel_xml_path=xml_path, verbose=False)
    retargeter = MDMToHumanoidRetargeter(env)
    retargeter.retarget_g1_mujoco(
        motion_data, 
        filename=output_file, 
        show_target_spheres=show_target_spheres, 
        position_offset=position_offset,
        scale=scale, 
        rotation=rotation
    )
    print("\n🎉 G1 리타게팅 완료!")


def meta_run(
    motions = '../ex-MoMo/save/20250724_1224',
):
    motion_dirs = os.listdir(motions)
    for motion_dir in motion_dirs:
        motion_file = os.path.join(motions, motion_dir, 'results.npy')
        retarget_motion(motion_file=motion_file, output_file=os.path.join(motions, motion_dir, f'{motion_dir}.mp4'))


if __name__ == "__main__":
    import fire
    fire.Fire(retarget_motion)