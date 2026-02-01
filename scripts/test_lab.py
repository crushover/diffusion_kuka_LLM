import mujoco
import mujoco.viewer
import numpy as np
import os
import time
import sys

# 导入控制器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from osc_controller import OSCController

def main():
    # 1. 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 指向带有夹爪、桌子和相机的最终场景
    xml_path = os.path.join(current_dir, "../assets/kuka_med7/scene_manipulation.xml")

    if not os.path.exists(xml_path):
        print(f"❌ 错误: 找不到文件 {xml_path}")
        return

    # 2. 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 3. 初始化控制器
    # 注意: 控制器只负责 A1-A7 关节，夹爪暂时不管
    try:
        controller = OSCController(model, data)
    except Exception as e:
        print(e)
        return

    # 4. 设置初始状态
    data.qpos[controller.dof_ids] = controller.q0
    mujoco.mj_forward(model, data)

    # 5. Mocap 归位
    mocap_id = model.body("target").mocapid[0]
    data.mocap_pos[mocap_id] = data.site(controller.site_id).xpos
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(controller.site_id).xmat)

    print("\n" + "="*50)
    print("������ Kuka Med 7 + Robotiq 2F-85 场景测试")
    print("������ 请在 Viewer 中检查：")
    print("   1. 机械臂是否装上了黑色夹爪？")
    print("   2. 前方是否有桌子？")
    print("   3. 按 Tab 键切换相机，查看 HandEye 和 SideCamera 视角")
    print("������ 双击红色方块拖动进行控制")
    print("="*50 + "\n")

    # 6. 仿真循环
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 初始视角设为侧后方，方便看全局
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.5, 0, 0.5]

        while viewer.is_running():
            step_start = time.time()

            # A. 获取输入
            target_pos = data.mocap_pos[mocap_id]
            target_quat = data.mocap_quat[mocap_id]

            # B. 计算力矩 (手臂)
            tau = controller.get_torque(target_pos, target_quat)

            # C. 发送指令
            # 1. 手臂控制
            ctrl_range = model.actuator_ctrlrange[controller.actuator_ids]
            tau = np.clip(tau, ctrl_range[:, 0], ctrl_range[:, 1])
            data.ctrl[controller.actuator_ids] = tau
            
            # 2. 夹爪控制 (暂时保持张开)
            # 找到夹爪的 actuator ID
            gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
            if gripper_actuator_id != -1:
                # 0 = Open, 255 = Closed
                data.ctrl[gripper_actuator_id] = 0.0 

            # D. 步进
            mujoco.mj_step(model, data)
            viewer.sync()

            # 保持实时性
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()