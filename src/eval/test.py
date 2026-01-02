import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch
import re
import random
import json
from tqdm import tqdm

device = "cuda:0"
seed = 42
set_seed(seed)
MODEL_PATH = "/HOME/paratera_xy/pxy1092/HDD_POOL/Q-Insight/src/open-r1-multimodal/output"
SUBFOLDER = "score-and-dist-v4"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
    subfolder=SUBFOLDER
)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, subfolder=SUBFOLDER)
template = "First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
SCORE_QUESTION_PROMPT = 'What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. Return the final answer with only one score in <answer> </answer> tags.'
suffix = "Return the final answer in <answer> </answer> tags."

def predict_one(image_path, question=SCORE_QUESTION_PROMPT):

    image_path = image_path

    # print(question + suffix)
    message = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question + suffix
                    # "text": "Describe this image."
                },
                {"type": "image", "image": f"file://{image_path}"}
            ]
        }
    ]

    text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = process_vision_info([message])
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    gen_config = GenerationConfig(
      do_sample=True,
      temperature=1.0,
      top_k=50,
      top_p=0.95,
      max_new_tokens=1024,
    )

    generated_ids = model.generate(
      **inputs,
      generation_config=gen_config,
      use_cache=True,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

def extract_think_score(img_path, text):
    reasoning = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    reasoning = reasoning[-1].strip()

    try:
        model_output_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        model_answer = model_output_matches[-1].strip() if model_output_matches else text.strip()

        # 数字结果的话再做一步处理
        score = float(re.search(r'\d+(\.\d+)?', model_answer).group())

        # 文字结果的话直接赋值
        # score = model_answer
    except:
        print(f"Meet error with {img_path}, please generate again.")
        score = random.randint(1, 5)

    return reasoning, score

def test_uwiqa():
    image_paths = os.listdir("/HOME/paratera_xy/pxy1092/HDD_POOL/Q-Insight/UWIQA/data/")
    # print(image_paths)

    output = {}
    for img_path in tqdm(image_paths):
        text = predict_one("/HOME/paratera_xy/pxy1092/HDD_POOL/Q-Insight/UWIQA/data/" + img_path)
        think, score = extract_think_score(img_path, text)
        output[img_path] = score

    with open("qinsightv4_uwiqa_result.json", 'w') as f:
        json.dump(output, f, indent=1)

def test_dist():
    with open("../test.json", 'r') as f:
        data = json.load(f)
    
    for sample in tqdm(data[: 1000]):
        if sample["question_type"] > 2:
            break
        image_path = "/DataB/mjx/fine_tune/M_Database/" + sample["image_name"]
        question = 'Please answer the question about underwater images and provide the answers from the given options.\n' + f'question: {sample["question"]}\ncandidate_answers: {"/".join(sample["candidate_answers"])}\n' + 'Please provide answer from the candidate_answers.'
        text = predict_one(image_path, question)
        think, answer = extract_think_score(image_path, text)
        sample["model_output"] = answer
    
    with open("qinsight_muiqd_qa_result.json", 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def test_desc():
    pass


if __name__ == '__main__':
    

    # with open("test_desc.json", 'w') as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)

    # text = predict_one("../M_Database/1.jpg")
    # print(text)
    # print(extract_think_score("../M_Database/1.jpg", text))

    test_uwiqa()