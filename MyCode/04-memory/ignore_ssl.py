"""
解决 SSL 证书验证失败的多种方法
注意：禁用 SSL 验证仅适用于开发/测试环境，生产环境请勿使用
"""

import requests
import urllib3
import ssl
import os

# 方法1: 全局禁用 urllib3 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 针对 LangChain trimmer.invoke() 的 SSL 配置
def configure_ssl_for_langchain():
    """配置 LangChain 相关的 SSL 设置"""
    # 设置环境变量禁用 SSL 验证
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''

    # 全局 SSL 上下文配置
    ssl._create_default_https_context = ssl._create_unverified_context
    print("已配置 LangChain SSL 设置")

def create_ssl_session():
    """创建禁用 SSL 验证的 requests 会话"""
    session = requests.Session()
    session.verify = False
    session.trust_env = False
    return session

def method1_requests_disable_ssl():
    """使用 requests 库禁用 SSL 验证"""
    try:
        response = requests.get('https://httpbin.org/get', verify=False)
        print("方法1 成功:", response.status_code)
        return response.json()
    except Exception as e:
        print("方法1 失败:", str(e))

def method2_urllib3_disable_ssl():
    """使用 urllib3 禁用 SSL 验证"""
    try:
        http = urllib3.PoolManager(cert_reqs='CERT_NONE')
        response = http.request('GET', 'https://httpbin.org/get')
        print("方法2 成功:", response.status)
        return response.data.decode('utf-8')
    except Exception as e:
        print("方法2 失败:", str(e))

def method3_custom_ssl_context():
    """创建自定义 SSL 上下文"""
    try:
        # 创建不验证证书的 SSL 上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        import urllib.request
        response = urllib.request.urlopen('https://httpbin.org/get', context=ssl_context)
        print("方法3 成功:", response.getcode())
        return response.read().decode('utf-8')
    except Exception as e:
        print("方法3 失败:", str(e))

def method4_langchain_trimmer_fix():
    """专门解决 trimmer.invoke() SSL 问题"""
    try:
        # 配置 SSL 设置
        configure_ssl_for_langchain()

        # 示例：如果你有 trimmer 对象，可以这样使用
        # trimmer.invoke(your_input)

        print("方法4: LangChain SSL 配置完成")
        print("现在可以尝试使用 trimmer.invoke() 方法")

    except Exception as e:
        print("方法4 配置失败:", str(e))

def recommended_solution():
    """推荐的解决方案 - 更新证书"""
    print("\n=== 推荐解决方案 ===")
    print("1. macOS 用户运行: /Applications/Python 3.x/Install Certificates.command")
    print("2. 更新 certifi: pip install --upgrade certifi")
    print("3. 更新 requests: pip install --upgrade requests")
    print("4. 如果使用公司网络，可能需要配置代理或证书")

if __name__ == "__main__":
    print("测试不同的 SSL 处理方法:\n")

    # 首先配置 LangChain SSL 设置
    method4_langchain_trimmer_fix()

    # 测试各种方法
    method1_requests_disable_ssl()
    method2_urllib3_disable_ssl()
    method3_custom_ssl_context()

    # 显示推荐解决方案
    recommended_solution()

    print("\n警告: 禁用 SSL 验证会带来安全风险，仅在开发环境使用!")
    print("\n特别提示: 如果 trimmer.invoke() 仍然失败，请检查:")
    print("1. 网络代理设置")
    print("2. 防火墙配置")
    print("3. 尝试重启 Python 环境")
