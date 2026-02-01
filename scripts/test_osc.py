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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "../assets/kuka_med7/scene_kuka.xml")

    if not os.path.exists(xml_path):
        print(f"❌ 错误: 找不到文件 {xml_path}")
        return


    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)


    try:
        controller = OSCController(model, data)
    except Exception as e:
        print(e)
        return


    data.qpos[controller.dof_ids] = controller.q0
    mujoco.mj_forward(model, data)


    mocap_id = model.body("target").mocapid[0]
    data.mocap_pos[mocap_id] = data.site(controller.site_id).xpos
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(controller.site_id).xmat)

    print("\n" + "="*50)
    print("Kuka Med 7 OSC 测试 (Extract Version)")
    print("双击红色方块，右键拖动")
    print("="*50 + "\n")


    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [0.5, 0, 0.5]

        while viewer.is_running():
            step_start = time.time()


            target_pos = data.mocap_pos[mocap_id]
            target_quat = data.mocap_quat[mocap_id]


            tau = controller.get_torque(target_pos, target_quat)


            ctrl_range = model.actuator_ctrlrange[controller.actuator_ids]
            tau = np.clip(tau, ctrl_range[:, 0], ctrl_range[:, 1])
            data.ctrl[controller.actuator_ids] = tau


            mujoco.mj_step(model, data)
            viewer.sync()


            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()