from typing import Any
from pathlib import Path
import json
from utils_and_prepare import time_decorator

class textGenerate:
    def __init__(self, args) -> None:
        import transformers, torch
        if not Path(args.model_name_or_path).exists():
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(repo_id=args.model_name_or_path, repo_type='model', local_files_only=True)
        else:
            model_path = args.model_name_or_path
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
        if "lora" in model_path.lower():
            print("load lora adapter")
            from peft import PeftModelForCausalLM
            adapter_config_json_path = Path(model_path)/"adapter_config.json"
            adapter_config = json.loads(adapter_config_json_path.read_text())
            # 移除unsloth训练后lora config里面多出来的键值
            for key in ["eva_config", "exclude_modules", "layer_replication",
                        "lora_bias", "use_dora", "use_rslora"]:
                if key in adapter_config:
                    adapter_config.pop(key)
            adapter_config_json_path.write_text(json.dumps(adapter_config, ensure_ascii=False, indent=1))
            peft_model = PeftModelForCausalLM.from_pretrained(self.pipeline.model, model_path)
            peft_model.set_adapter("default")
            self.pipeline.model = peft_model
    def __call__(self, conversation, do_sample=False, *args, **kwds):
        outputs = self.pipeline(conversation, do_sample = do_sample, max_new_tokens=512)
        response = outputs[0]["generated_text"][-1]["content"]
        return response

class zhipuGenerate:
    # pip install zhipuai
    def __init__(self, args) -> None:
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key="replace with your key")
    def __call__(self, conversation, *args: Any, **kwds: Any) -> Any:
        try:
            response = self.client.chat.completions.create(
                model="glm-4-voice",
                messages=conversation,
                top_p= 0.95,
                temperature= 0.95,
                max_tokens=1024,
                stream=False
            )
            output = response.choices[0].message.content
        except Exception as e:
            print(f"glm-4-voice error with {e}")
            output = ""
        return output

class openaiGptGenerate:
    # pip install openai
    def __init__(self, args) -> None:
        import httpx
        from openai import OpenAI
        http_client = httpx.Client(
            transport=httpx.HTTPTransport(local_address="0.0.0.0"))
        client = OpenAI(
            api_key="sk-proj-replace with your key",
            http_client=http_client
        )
        self.client = client
        self.model_id = args.model_name_or_path
    def __call__(self, conversation, *args: Any, **kwds: Any) -> Any:
        completion = self.client.chat.completions.create(
            model=self.model_id,
            modalities=["text"],
            # audio={"voice": "alloy", "format": "wav"},
            messages=conversation)
        output = completion.choices[0].message.content
        return output

class qwenOmniGenerate:
    # pip install openai
    def __init__(self, args) -> None:
        import httpx
        from openai import OpenAI
        http_client = httpx.Client(
            transport=httpx.HTTPTransport(local_address="0.0.0.0"))
        client = OpenAI(
            api_key="replace with your key",
            http_client=http_client,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.client = client
        self.model_id = args.model_name_or_path
    def __call__(self, conversation, *args: Any, **kwds: Any) -> Any:
        completion = self.client.chat.completions.create(
            model=self.model_id,
            modalities=["text"],
            # audio={"voice": "alloy", "format": "wav"},
            messages=conversation,
            stream=True,
            stream_options={"include_usage": True})
        response = ""
        for chunk in completion:
            if chunk.choices:
                delta_content = chunk.choices[0].delta.content
                if isinstance(delta_content, str):
                    response += delta_content
        return response

"""
apt install ffmpeg
pip install torch transformers peft accelerate --upgrade
pip install soundfile librosa backoff
export CUDA_HOME=/mnt/publiccache/zxc/cuda-12.3/
pip install flash-attn --no-build-isolation
"""
class phi4multimodalGenerate:
    def __init__(self, args) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        import torch
        if not Path(args.model_name_or_path).exists():
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(repo_id=args.model_name_or_path, repo_type='model', local_files_only=True)
        else:
            model_path = args.model_name_or_path
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            device_map="cuda", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            # if you do not use Ampere or later GPUs, change attention to "eager"
            _attn_implementation='flash_attention_2', # save memory and efficient
        ).cuda()
        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(model_path)

    def __call__(self, conversation, do_sample=False, *args, **kwds):
        # conversation == messages
        # content not list
        conversation_, audio_paths = list(), list()
        for message in conversation:
            if not isinstance(message["content"], str):
                audio_paths.append(message["content"]["input_audio"]["data"])
                message ["content"] = f"<|audio_{len(audio_paths)}|>"
            conversation_.append(message)
        import soundfile as sf
        audios = [sf.read(audio_path) for audio_path in audio_paths]
        prompt = self.processor.apply_chat_template(conversation_, add_generation_prompt=True)
        # Process with the model
        inputs = self.processor(text=prompt, audios=audios if audios else None, return_tensors='pt').to('cuda:0')
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=300,
            generation_config=self.generation_config,
            num_logits_to_keep = 0,
            do_sample=do_sample
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

class qwen2AudioGenerate:
    def __init__(self, args) -> None:
        from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioConfig
        from transformers.models.qwen2_audio.processing_qwen2_audio import Qwen2AudioProcessor
        import torch
        if not Path(args.model_name_or_path).exists():
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(repo_id=args.model_name_or_path, repo_type='model', local_files_only=True)
        else:
            model_path = args.model_name_or_path
        # Load model and processor
        self.processor = Qwen2AudioProcessor.from_pretrained(args.model_name_or_path)
        self.config = Qwen2AudioConfig.from_pretrained(args.model_name_or_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16, 
            # if you do not use Ampere or later GPUs, change attention to "eager"
            # _attn_implementation='flash_attention_2', # save memory and efficient
            device_map="auto",)
    def __call__(self, conversation, do_sample=False, *args, **kwds):
        from io import BytesIO
        import librosa
        from urllib.request import urlopen
        def url_or_path2bytes(url_or_path:str):
            audio_bytes = urlopen(url_or_path).read() if url_or_path.startswith("http") \
                else open(url_or_path, "rb").read()
            return audio_bytes
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if "audio" in ele["type"]:
                        ele["type"] = "audio"
                        ele['audio_url'] = ele["local_file"]
                        audio_bytes = url_or_path2bytes(ele['audio_url'])
                        audios.append(librosa.load(
                            BytesIO(audio_bytes),
                            sr=self.processor.feature_extractor.sampling_rate)[0]
                        )
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # ids = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True)
        inputs = self.processor(text=text, audios=audios if audios else None, return_tensors="pt", padding=True)
        inputs.input_ids = inputs.input_ids.to("cuda")
        self.model.generation_config.do_sample = do_sample
        generate_ids = self.model.generate(**inputs, max_new_tokens=512)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response


"""

pip install qwen-omni-utils[decord]
apt install ffmpeg
pip install torch transformers peft accelerate torchvision --upgrade
pip install soundfile librosa backoff
export CUDA_HOME=/mnt/publiccache/zxc/cuda-12.3/
pip install flash-attn --no-build-isolation
pip install openai
"""
class Qwen2_5OmniGenerate:
    def __init__(self, args) -> None:
        from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor, Qwen2_5OmniConfig
        import torch
        # Load model and processor
        # We recommend enabling flash_attention_2 for better acceleration and memory saving.
        self.model = Qwen2_5OmniModel.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            # if you do not use Ampere or later GPUs, change attention to "eager"
            attn_implementation='flash_attention_2', # save memory and efficient
            enable_audio_output=False
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name_or_path)
        self.config = Qwen2_5OmniConfig.from_pretrained(args.model_name_or_path)
    @time_decorator
    def __call__(self, conversation, do_sample=False, *args, **kwds):
        for ele in conversation:
            if isinstance(ele["content"], list):
                for kv in ele["content"]:
                    if "audio" in kv["type"]:
                        kv["type"] = "audio"
                        if "audio" not in kv:
                            kv["audio"] = kv["local_file"]
        from qwen_omni_utils import process_mm_info
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        output = self.model.generate(**inputs, use_audio_in_video=True, return_audio=False, do_sample=do_sample)
        texts = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return texts[0].split("assistant\n")[-1]

def test_zhipu():
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hi"
                },
                {
                    "type": "text",
                    "text": "what is the weather?"
                }
                # {
                #     "type": "input_audio",
                #     "input_audio": {
                #         "data": "<base64_string>",
                #         "format":"wav"
                #     }
                # }
            ]
        },
    ]
    zhipu_generate = zhipuGenerate()
    output = zhipu_generate(conversation=messages)
    print(output)

def test_qwen25():
    messages = [
            {"role": "user", "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "audio", "audio": "InterviewGuideVoice.mp3"},
                    {"type": "audio", "audio": "depression_instances/wav-A（73例）/001A/0.wav"}
                ]
            },
        ]
    from types import SimpleNamespace
    args = SimpleNamespace(model_name_or_path="Qwen/Qwen2.5-Omni-7B")
    qwen2_5OmniGenerate = Qwen2_5OmniGenerate(args=args)
    text = qwen2_5OmniGenerate(conversation=messages)

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).absolute().parent)
    test_qwen25()
