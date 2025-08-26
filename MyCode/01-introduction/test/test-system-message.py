import inspect
from langchain_core.messages import HumanMessage, SystemMessage
print(SystemMessage)                 # 查看类表示
print(SystemMessage.__module__)      # 模块名
help(SystemMessage)                  # 文档/构造签名
print(inspect.getsource(SystemMessage))   # 如果是 Python 实现，会打印源码