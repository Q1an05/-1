# 数模模拟1

## 环境配置与注意事项

1. **推荐Python版本**：建议使用 Python 3.8 及以上版本。

2. **依赖库安装**：
   - 本项目主要依赖 `pandas` 库进行数据处理和 Excel 文件读取。
   - 安装依赖建议使用如下命令：
     ```bash
     pip install pandas
     ```
   - 如需处理 `.xlsx` 文件，`pandas` 会自动调用 `openpyxl`，如遇到相关报错可手动安装：
     ```bash
     pip install openpyxl
     ```

3. **虚拟环境建议**：
   - 推荐使用虚拟环境隔离依赖，避免与系统环境冲突。
   - 创建虚拟环境示例：
     ```bash
     python -m venv venv
     source venv/bin/activate  # macOS/Linux
     venv\Scripts\activate    # Windows
     ```

4. **中文路径与文件名注意事项**：
   - 本项目包含中文路径和中文文件名，建议确保操作系统和 Python 环境对中文路径支持良好。
   - 如遇文件找不到或编码问题，请检查文件路径、文件名及当前工作目录。

5. **运行脚本**：
   - 进入根目录后运行主程序：
     ```bash
     python main.py
     ```

6. **其他依赖**：
   - 如有其他依赖库，请根据实际需求补充安装。
