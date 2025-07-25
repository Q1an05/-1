import scripts.resource_processor as rp


#excel文件路径
excel_file_path = 'resource/B题-支路车流量推测问题 附件(Attachment).xlsx'

#第一题
df1 = rp.read_excel("表1 (Table 1)", excel_file_path)
