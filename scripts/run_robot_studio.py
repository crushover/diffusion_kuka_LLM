
import mujoco
import mujoco.viewer
import numpy as np
import os
import time
import sys
import cv2

# 导入 OSC 控制器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from osc_controller import OSCController

def main():
    # ---------------------------------------------------------
    # 1. 场景设置
    # ---------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "../assets/kuka_med7/scene_manipulation.xml")

    if not os.path.exists(xml_path):
        print(f"❌ 错误: 找不到文件 {xml_path}")
        return

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # ---------------------------------------------------------
    # 2. 初始化控制器与ID
    # ---------------------------------------------------------
    try:
        controller = OSCController(model, data)
    except Exception as e:
        print(f"❌ 控制器报错: {e}")
        return
    
    # 【修复点】在这里获取 mocap_id，让它在整个 main 函数里都可用
    try:
        mocap_id = model.body("target").mocapid[0]
    except KeyError:
        print("❌ 错误: XML里找不到 name='target' 的 body！")
        return

    # 定义重置函数
    def reset_sim():
        # 1. 重置关节角度
        data.qpos[controller.dof_ids] = controller.q0
        
        # 2. 刷新运动学
        mujoco.mj_forward(model, data)

        # 3. Mocap 归位 (直接使用外面定义的 mocap_id)
        data.mocap_pos[mocap_id] = data.site(controller.site_id).xpos
        mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(controller.site_id).xmat)
        
        # 4. 如果有方块，也重置方块位置 (可选)
        # 找到方块的关节地址
        cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        if cube_joint_id != -1:
            # 获取该关节在 qpos 中的地址
            qpos_adr = model.jnt_qposadr[cube_joint_id]
            # 重置方块位置 (x=0.6, y=0, z=0.15) 和 姿态 (1,0,0,0)
            data.qpos[qpos_adr:qpos_adr+3] = [0.6, 0, 0.15]
            data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

        print("检测到重置：已恢复初始状态")

    # 首次运行初始化
    reset_sim()

    # ---------------------------------------------------------
    # 3. 初始化视觉系统
    # ---------------------------------------------------------
    renderer = mujoco.Renderer(model, height=480, width=640)
    cameras = ["hand_camera", "side_camera", "top_camera"]
    render_every_n_steps = 15 
    step_counter = 0

    print("\n" + "="*60)
    print("机械臂全能指挥台 (Robot Studio)")
    print("="*60 + "\n")

    # ---------------------------------------------------------
    # 4. 主循环
    # ---------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.8
        viewer.cam.lookat[:] = [0.6, 0, 0.4]

        while viewer.is_running():
            loop_start = time.time()
            step_counter += 1

            # === A. 重置看门狗 ===
            if data.time < 0.001:
                reset_sim()

            # === B. 物理控制 ===
            # 这里现在能正确访问 mocap_id 了
            target_pos = data.mocap_pos[mocap_id]
            target_quat = data.mocap_quat[mocap_id]
            
            tau = controller.get_torque(target_pos, target_quat)
            
            ctrl_range = model.actuator_ctrlrange[controller.actuator_ids]
            tau = np.clip(tau, ctrl_range[:, 0], ctrl_range[:, 1])
            data.ctrl[controller.actuator_ids] = tau
            
            # 夹爪控制 (按下空格键闭合，松开张开 - 简单的键盘交互示例)
            # 这里先保持张开，录数据时我们会加按钮
            gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
            if gripper_id != -1: 
                data.ctrl[gripper_id] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            # === C. 相机渲染 ===
            if step_counter % render_every_n_steps == 0:
                for cam_name in cameras:
                    renderer.update_scene(data, camera=cam_name)
                    pixels = renderer.render()
                    img_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                    cv2.putText(img_bgr, cam_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(cam_name, img_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time_until_next_step = model.opt.timestep - (time.time() - loop_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()