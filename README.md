## 项目更新指南 (Git Workflow)

如果我在本地修改了代码，需要执行以下步骤同步到 GitHub：

1.  **查看状态 (检查修改了哪些文件)**
    ```bash
    git status
    ```
    *确认红色的文件列表是你想要上传的修改。*

2.  **添加到暂存区**
    ```bash
    git add .   "上传全部的代码"


    git add train.py utils/robot_control.py "选择上传的代码"
    ```

3.  **提交更改 (打上版本标签)**
    ```bash
    git commit -m "在这里写下你这次具体修改了什么功能"
    ```
    *例如：git commit -m "优化了机械臂抓取逻辑，修复了IK报错"*

4.  **推送到远程仓库**
    ```bash
    git push
    ```
    *看到 "Everything up-to-date" 或进度条走完即成功。*