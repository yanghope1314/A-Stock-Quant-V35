#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感模型离线下载脚本（国内服务器专用）
====================================================
功能：从 hf-mirror.com 下载 uer/roberta-base-finetuned-jd-binary-chinese
      下载一次后本地永久缓存，NLP引擎启动时直接读本地，不再访问网络

使用方法：
  python download_sentiment_model.py

  或指定保存目录：
  python download_sentiment_model.py --save-dir /data/models/roberta-jd-binary

  下载后在 config_v19.py 中设置（可选，不设置也能自动找到）：
  NLP_CONFIG = {
      ...
      'bert_model_path': '/data/models/roberta-jd-binary',
  }

常见问题：
  Q: 提示 SSLError 或 Connection refused?
  A: 确认服务器能访问 hf-mirror.com，可以先 curl https://hf-mirror.com

  Q: 下载中断了怎么办?
  A: 重新运行，--resume-download 会从断点继续

  Q: 需要多少磁盘空间?
  A: 约 400MB（RoBERTa Base + 分类头权重）
====================================================
"""
import os
import sys
import argparse

# ── 必须在 import transformers 之前设置镜像 ──────────────────────────────────
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODEL_ID   = 'uer/roberta-base-finetuned-jd-binary-chinese'
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'roberta-jd-binary')


def download_with_huggingface_cli(save_dir: str) -> bool:
    """方法1：使用 huggingface-cli（最可靠，支持断点续传）"""
    import subprocess
    print("\n📦 方法1: huggingface-cli 下载（支持断点续传）")
    cmd = [
        sys.executable, '-m', 'huggingface_hub.commands.huggingface_cli',
        'download', MODEL_ID,
        '--local-dir', save_dir,
        '--resume-download',
    ]
    env = os.environ.copy()
    env['HF_ENDPOINT'] = 'https://hf-mirror.com'

    try:
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  huggingface-cli 失败: {e}")
        return False
    except FileNotFoundError:
        print("  huggingface-cli 不可用（未安装 huggingface_hub）")
        return False


def download_with_snapshot(save_dir: str) -> bool:
    """方法2：使用 snapshot_download API"""
    print("\n📦 方法2: snapshot_download API")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=save_dir,
            endpoint='https://hf-mirror.com',
            resume_download=True,
            ignore_patterns=['*.msgpack', 'flax_model*', 'tf_model*', 'rust_model*'],
        )
        return True
    except ImportError:
        print("  huggingface_hub 未安装: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  snapshot_download 失败: {e}")
        return False


def download_with_from_pretrained(save_dir: str) -> bool:
    """方法3：直接 from_pretrained + save_pretrained"""
    print("\n📦 方法3: from_pretrained 在线加载后保存")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"  正在从 hf-mirror.com 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print(f"  正在从 hf-mirror.com 加载模型（约400MB，请耐心等待）...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        print(f"  ✅ 已保存到: {save_dir}")
        return True
    except ImportError:
        print("  transformers 未安装: pip install transformers torch")
        return False
    except Exception as e:
        print(f"  from_pretrained 失败: {e}")
        return False


def verify_model(save_dir: str) -> bool:
    """验证下载的模型可以正确加载和推理"""
    print(f"\n🔍 验证模型: {save_dir}")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(save_dir, local_files_only=True)
        model.eval()

        # 测试推理
        test_cases = [
            ("今天大涨，利好消息不断，公司业绩超预期", "正面"),
            ("连续下跌，主力出货，利空不断", "负面"),
        ]
        print("  推理测试：")
        import torch.nn.functional as F
        for text, expected in test_cases:
            inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze()
            label = "正面" if probs[1] > probs[0] else "负面"
            score = float(probs[1].item() - probs[0].item())
            status = "✅" if label == expected else "⚠️"
            print(f"  {status} \"{text[:20]}...\" → {label}（得分: {score:+.3f}，期望: {expected}）")

        print(f"\n✅ 验证通过！模型可正常使用")
        return True
    except Exception as e:
        print(f"  ❌ 验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='下载中文情感分析模型（国内镜像）')
    parser.add_argument('--save-dir', default=DEFAULT_DIR,
                        help=f'模型保存目录（默认: {DEFAULT_DIR}）')
    parser.add_argument('--verify-only', action='store_true',
                        help='只验证已下载的模型，不重新下载')
    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    print(f"{'='*60}")
    print(f"📥 中文情感模型下载脚本")
    print(f"{'='*60}")
    print(f"  模型ID:  {MODEL_ID}")
    print(f"  保存目录: {save_dir}")
    print(f"  镜像站:  {os.environ['HF_ENDPOINT']}")
    print(f"{'='*60}")

    if args.verify_only:
        verify_model(save_dir)
        return

    # 检查是否已经存在
    if os.path.isdir(save_dir) and any(f.endswith('.json') for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))):
        print(f"\n⚡ 目录已存在且含有配置文件，先验证...")
        if verify_model(save_dir):
            print("\n模型已就绪，无需重新下载。")
            print(f"在 config_v19.py 中配置（可选）：")
            print(f"  NLP_CONFIG['bert_model_path'] = '{save_dir}'")
            return
        print("  验证失败，重新下载...")

    os.makedirs(save_dir, exist_ok=True)

    # 依次尝试三种方法
    success = False
    for method in [download_with_huggingface_cli, download_with_snapshot, download_with_from_pretrained]:
        if method(save_dir):
            success = True
            break

    if not success:
        print(f"\n{'='*60}")
        print("❌ 所有自动下载方法均失败")
        print("\n手动下载方法：")
        print("  1. 在有网络的机器上下载：")
        print(f"     export HF_ENDPOINT=https://hf-mirror.com")
        print(f"     pip install huggingface_hub")
        print(f"     huggingface-cli download {MODEL_ID} --local-dir ./roberta-jd-binary")
        print(f"  2. 将 ./roberta-jd-binary/ 文件夹上传到服务器的 {save_dir}")
        print(f"  3. 在 config_v19.py 设置 NLP_CONFIG['bert_model_path'] = '{save_dir}'")
        print(f"{'='*60}")
        sys.exit(1)

    # 验证
    if verify_model(save_dir):
        print(f"\n{'='*60}")
        print("🎉 模型下载并验证成功！")
        print(f"\n配置说明（可选，不配置也会自动找到）：")
        print(f"  在 config_v19.py 中添加：")
        print(f"  NLP_CONFIG['bert_model_path'] = '{save_dir}'")
        print(f"\n或设置环境变量（推荐写入 ~/.bashrc 永久生效）：")
        print(f"  export SENTIMENT_MODEL_DIR={save_dir}")
        print(f"{'='*60}")
    else:
        print("\n⚠️ 下载完成但验证失败，请检查文件完整性后重试")
        sys.exit(1)


if __name__ == '__main__':
    main()