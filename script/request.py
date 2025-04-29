import sys
import os
import json
from openai import OpenAI

with open('api.key', 'r') as f:
    API_KEY = f.read()

def format_request(path, text, f) :
    request = {
        "custom_id": path,
        "body": {
            "messages": [
                {"role": "system", "content": "对 user 给出的内容进行摘要。你的输出将会被直接使用，所以不要有任何额外的欢迎词，要保持用语的严谨、专业，除了摘要不要有多余内容。摘要应该简洁，纯文本格式，禁止包括代码！"}, 
                {"role": "user", "content": text}
            ]
        }
    }
    print(json.dumps(request), file=f)

def gen_requests(docs_dir) :
    with open("request.jsonl", 'w') as output_file:
        for subject_name in os.listdir(docs_dir):
            subject_path = os.path.join(docs_dir, subject_name)
            if not os.path.isdir(subject_path) :
                continue

            for file in os.listdir(subject_path):
                if not file.endswith('.md') :
                    continue

                file_path = os.path.join(subject_path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    path = os.path.join(subject_name, file)
                    format_request(path, text[:7500], output_file)

def make_request() :
    client = OpenAI(
        api_key=API_KEY, 
        base_url="https://api.siliconflow.cn/v1"
    )

    batch_input_file = client.files.create(
        file=open("request.jsonl", "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.data["id"]

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "oi-wiki"
        },
        extra_body={"replace":{"model": "deepseek-ai/DeepSeek-V3"}}
    )

    return batch


def main(
        docs_dir : str = './OI-wiki/docs/',
    ) :
    gen_requests(docs_dir)
    batch = make_request()
    return batch

if __name__ == "__main__" :
    batch = main(*sys.argv[1:])
    print(batch)
