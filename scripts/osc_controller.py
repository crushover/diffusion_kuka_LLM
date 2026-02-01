import mujoco
import numpy as np
from typing import Optional, List

class OSCController:
    """
    基于 kuka_osc_tracking.py 提取的控制器。
    保留所有原始参数和计算逻辑。
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # ==========================================
        # 1. 原始控制参数 (完全保留)
        # ==========================================
        self.impedance_pos = np.asarray([75.0, 75.0, 75.0])  # [N/m]
        self.impedance_ori = np.asarray([20.0, 20.0, 20.0])  # [Nm/rad]
        
        # 关节零空间刚度
        self.Kp_null = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
        
        self.damping_ratio = 1.0
        self.Kpos = 0.95
        self.Kori = 0.95
        self.integration_dt = 1.0
        self.gravity_compensation = True

        # 计算阻尼矩阵 (初始化时计算一次即可)
        self.damping_pos = self.damping_ratio * 2 * np.sqrt(self.impedance_pos)
        self.damping_ori = self.damping_ratio * 2 * np.sqrt(self.impedance_ori)
        self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        self.Kd = np.concatenate([self.damping_pos, self.damping_ori], axis=0)
        self.Kd_null = self.damping_ratio * 2 * np.sqrt(self.Kp_null)

        # ==========================================
        # 2. 获取 ID 和 索引
        # ==========================================
        self.site_name = "attachment_site"
        try:
            self.site_id = model.site(self.site_name).id
            self.joint_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
            self.dof_ids = np.array([model.joint(name).id for name in self.joint_names])
            self.actuator_ids = np.array([model.actuator(name).id for name in self.joint_names])
        except KeyError as e:
            raise ValueError(f"❌ 初始化失败: 无法在模型中找到组件 {e}")

        # 记录初始姿态 q0 (用于零空间控制)
        self.q0 = np.array([0, 0.6, 0, -1.5, 0, 1.0, 0])
        
        # ==========================================
        # 3. 内存预分配 (完全保留)
        # ==========================================
        self.jac_full = np.zeros((6, model.nv))
        self.M_full = np.zeros((model.nv, model.nv))
        
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        
        # 机械臂自由度
        self.n_arm = len(self.dof_ids)

        print("✅ OSCController (Tracking Version) 初始化完成")

    def get_torque(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """
        计算控制力矩
        逻辑完全对应 kuka_osc_tracking.py 的 while 循环内部
        """
        # 1. 计算 Twist (位置和姿态误差)
        # 位置误差
        dx = target_pos - self.data.site(self.site_id).xpos
        self.twist[:3] = self.Kpos * dx / self.integration_dt
        
        # 姿态误差 (使用原始的 mju_mulQuat 逻辑)
        mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= self.Kori / self.integration_dt

        # 2. 计算并截取雅可比 (Jacobian)
        mujoco.mj_jacSite(self.model, self.data, self.jac_full[:3], self.jac_full[3:], self.site_id)
        J_arm = self.jac_full[:, self.dof_ids]

        # 3. 计算并截取质量矩阵 (Inertia Matrix)
        mujoco.mj_fullM(self.model, self.M_full, self.data.qM)
        M_arm = self.M_full[np.ix_(self.dof_ids, self.dof_ids)]
        
        # 求逆 (加微小阻尼)
        M_inv_arm = np.linalg.inv(M_arm + np.eye(self.n_arm) * 1e-6)

        # 4. 计算操作空间惯量矩阵 Mx
        Mx_inv = J_arm @ M_inv_arm @ J_arm.T
        
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        # 5. 计算力矩
        dq_arm = self.data.qvel[self.dof_ids]
        dx_vel = J_arm @ dq_arm # 当前笛卡尔速度
        
        # F = Mx * (Kp * error - Kd * velocity)
        # 注意：这里 twist 包含了 error 项
        tau = J_arm.T @ Mx @ (self.Kp * self.twist - self.Kd * dx_vel)

        # 6. 零空间控制
        Jbar = M_inv_arm @ J_arm.T @ Mx
        ddq = self.Kp_null * (self.q0 - self.data.qpos[self.dof_ids]) - self.Kd_null * dq_arm
        tau += (np.eye(self.n_arm) - J_arm.T @ Jbar.T) @ ddq

        # 7. 重力补偿
        if self.gravity_compensation:
            tau += self.data.qfrc_bias[self.dof_ids]
            
        return tau