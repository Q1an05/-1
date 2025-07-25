import pandas as pd

#这个脚本是用来读取excel文件的，并返回一个pandas的DataFrame
#sheet_name是excel文件的表名，file_path是excel文件的路径
def read_excel(sheet_name, file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"读取时发生错误: {e}")
        return None