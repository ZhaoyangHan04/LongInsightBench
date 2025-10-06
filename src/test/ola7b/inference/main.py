import os
os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "480"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'
import sys
sys.path.append('./')
import json
from tqdm import tqdm
import torch
import argparse

from ola.model.builder import load_pretrained_model
from ola.conversation import conv_templates
from ola.mm_utils import process_anyres_video
from ola.datasets.preprocess import (
    tokenizer_speech_image_token,
)
from ola.constants import (
    IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
)
from ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from ola.conversation import SeparatorStyle
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import moviepy.editor as mpy
# from moviepy import editor as mpy
import librosa
import whisper

USER_PROMPT = """
You are an expert in long video understanding. Always base your answers strictly on the video content.

Each question may have one or more correct answer(s). Please think step-by-step, and then output the **option label(s)** ('A','B','C','D') and a **brief explanation** that explain the reason for your choices.

question: {question}
options: {options}

Output format:
    "answer": [your choice(s)],
    "reason": your explanation
"""

# ========= 函数部分 =========
def load_audio(audio_file_name):
    speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
    if len(speech_wav.shape) > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000

    speechs, speech_wavs = [], []
    if len(speech_wav) <= CHUNK_LIM:
        speech = whisper.pad_or_trim(speech_wav)
        speech_wav = whisper.pad_or_trim(speech_wav)
        speechs.append(speech)
        speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i: i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))

    mels = []
    for chunk in speechs:
        chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
        mels.append(chunk)

    mels = torch.cat(mels, dim=0)
    speech_wavs = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 25:
        mels = mels[:25]
        speech_wavs = speech_wavs[:25]

    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs


def extract_audio(video_path):
    my_clip = mpy.VideoFileClip(video_path)
    return my_clip.audio


def ask_model(model, tokenizer, image_processor, video_path, text):
    """核心推理函数，输入视频和问题，返回模型答案"""

    modality = "video"
    visual = video_path

    # 抽帧
    vr = VideoReader(visual, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 64, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(frame) for frame in spare_frames]

    # 音频
    audio = extract_audio(visual)
    audio.write_audiofile("./video_audio.wav")
    video_audio_path = './video_audio.wav'
    speech, speech_length, speech_chunk, speech_wav = load_audio(video_audio_path)

    # 拼接 prompt
    qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + text
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_speech_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')

    # 视频预处理
    video_processed = []
    for idx, frame in enumerate(video):
        image_processor.do_resize = False
        image_processor.do_center_crop = False
        frame = process_anyres_video(frame, image_processor)
        video_processed.append(frame.unsqueeze(0))
    video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
    video_processed = (video_processed, video_processed)

    # 生成
    attention_masks = input_ids.ne(151643).long().to('cuda')
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=video_processed[0],
            images_highres=video_processed[1],
            modalities="video",
            speech=[speech.bfloat16().to('cuda')],
            speech_lengths=[speech_length.to('cuda')],
            speech_chunks=[speech_chunk.to('cuda')],
            speech_wav=[speech_wav.to('cuda')],
            attention_mask=attention_masks,
            stopping_criteria=[stopping_criteria],
            max_new_tokens=2048,
            temperature=0.3,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs.strip()


# ========= 主逻辑 =========
if __name__ == "__main__":
    MODEL_PATH = "./models/Ola-7b"  
    curren_tasks = ["1intra_event_reasoning", 
                    "2multimodal_temporal_localization", 
                    "3audio_visual_alignment", 
                    "4timeline_reconstruction", 
                    "5topic_stance_evolution_summarization", 
                    "6cross_event_causality"] 
    for curren_task in curren_tasks:
        print(f"===== 处理任务: {curren_task} =====")
        INPUT_FILE = f"./final_qa_subset/{curren_task}.json"
        OUTPUT_FILE = f"./experiment_frames/ola7b_raw/64/{curren_task}.json"
        VIDEO_ROOT = "./datasets/finevideo/videos"
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        # 加载模型
        tokenizer, model, image_processor, _ = load_pretrained_model(MODEL_PATH, None)
        model = model.to("cuda").eval().bfloat16()

        # 已有结果
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            results = {}

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        for qa in tqdm(qa_data):
            qid = qa["question_id"]
            question = qa["question"]
            video_id = qa["related_videoID"]
            options = qa.get("options", "")

            if qid in results:
                continue

            try:
                category, idx = video_id.rsplit("_", 1)
                video_path = os.path.join(VIDEO_ROOT, category, f"sample_{idx}.mp4")
            except Exception as e:
                print(f"⚠️ 无法解析 videoID: {video_id}, 跳过。异常: {e}")
                continue

            if not os.path.exists(video_path):
                print(f"⚠️ 视频不存在: {video_path}")
                continue

            # 拼接问题
            prompt = USER_PROMPT.format(question=question, options=options)

            # 调用 Ola 模型
            try:
                answer = ask_model(model, tokenizer, image_processor, video_path, prompt)
            except Exception as e:
                print(f"⚠️ 推理失败: {video_id}, 异常: {e}")
                continue

            results[qid] = {
                "question_id": qid,
                "question": question,
                "options": options,
                "video_id": video_id,
                "model_answer": answer
            }

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✅ 完成推理，结果已保存到 {OUTPUT_FILE}")
