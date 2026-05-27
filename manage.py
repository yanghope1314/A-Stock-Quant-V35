#!/usr/bin/env python
"""Django's command-line utility for administrative tasks & one-click launcher."""
import os
import sys
import threading
import time
import webbrowser
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def open_browser(port):
    # 等待服务器启动并打开浏览器（因为加载大量 AI 模型需要时间，将超时延长至 45 秒）
    print("⌛ 等待系统就绪...")
    for _ in range(45):
        if is_port_in_use(port):
            print("\n✨ 系统已就绪！正在打开浏览器界面...")
            webbrowser.open(f"http://127.0.0.1:{port}")
            break
        time.sleep(1)

def main():
    """Run administrative tasks or default startup sequence."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_project.settings')
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # 如果无参数运行，默认执行一键启动流程 (整合了原来的 start.py 逻辑)
    if len(sys.argv) == 1:
        print("==================================================")
        print("  A股量化选股系统 V35 OpenSource - 自动启动")
        print("==================================================")
        print(f"🐍 Python 版本: {sys.version.split()[0]}")
        
        # 1. 自动执行数据库迁移
        print("📦 正在检查并运行数据库迁移...")
        try:
            import django
            django.setup()
            from django.core.management import call_command
            call_command('migrate', interactive=False)
            print("✅ 数据库迁移检查完毕。")
        except Exception as e:
            print(f"⚠️ 数据库迁移提示: {e} (这通常不影响运行)")

        port = 8000
        if is_port_in_use(port):
            print(f"⚠️ 端口 {port} 已被占用，请确保没有其他实例在运行。")
        
        print(f"🌐 系统将运行在 http://127.0.0.1:{port}")
        
        # 2. 运行 Django 开发服务器 (--noreload 避免双进程，加速 AI 模型加载)
        import subprocess
        server_process = subprocess.Popen([sys.executable, sys.argv[0], "runserver", "--noreload", f"0.0.0.0:{port}"])
        
        # 3. 等待并打开浏览器
        open_browser(port)
        
        # 4. 等待子进程结束
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 正在停止系统...")
            server_process.terminate()
        sys.exit(0)
    
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
