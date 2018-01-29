date文件夹里的数据，是用正则表达式提取出来的，对提取出来的每道工序做差。

extract_date.py 提取原始数据中的时间
monitornoise.py 假设数据是符合正态分布，运用3σ原则对异常值进行处理，效果不太理想
Industry_AI _Preprocessing.py 数据预处理
model_select.py 模型选择