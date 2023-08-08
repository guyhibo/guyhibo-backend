from models.model10_transformer import AV_Transformer
import torch
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object

CKPT_PATH = "ckpt/custom_model10_30.pt"
VOCAB_CSV_PATH = "data/vocab.csv"
SENTENCE_EBD_PATH = "data/sentence_ebd7.pt"

def load_label(): # 음절마다의 id 값을 저장해놓은 pt파일을 로드
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(VOCAB_CSV_PATH, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"]

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char

def make_input_tensor(video_path): # 영상을 표준화 및 프레임을 550으로 맞춰서 텐서로 내뱉는 전처리

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.522], std=[0.125]),
        ])
    
    input_tensor_list = []
    
    cap = cv2.VideoCapture(video_path)
    total_frame_ct = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_pil = Image.fromarray(frame/255.0)
        input_tensor = preprocess(frame_pil)
        input_tensor_list.append(input_tensor)
    
    input_sequence = torch.stack(input_tensor_list)

    zeros_tensor = torch.zeros((550, 1, 96, 96))

    # 원본 입력 텐서를 0으로 채운 텐서와 결합하여 크기를 확장
    input_sequence = torch.cat([input_sequence, zeros_tensor[total_frame_ct:, :, :, :]], dim=0)
    del zeros_tensor
    torch.cuda.empty_cache()
    
    return input_sequence, total_frame_ct

def get_prediction(model_transformer, video_path): # 영상의 경로를 받아 예측 문장을 내뱉는 함수

    video_inputs, video_input_lengths = make_input_tensor(video_path)

    video_inputs = video_inputs.unsqueeze(0)
    video_inputs = video_inputs.transpose(1, 2)

    video_input_lengths = torch.tensor([video_input_lengths], dtype=torch.long)
    video_input_lengths = video_input_lengths.unsqueeze(0)

    video_inputs, video_input_lengths = video_inputs.to(device), video_input_lengths.to(device)

    with torch.no_grad():
        
        y_hats = model_transformer.recognize(video_inputs, video_input_lengths)
        y_hats = y_hats.squeeze(0)

    _, max_indices = torch.max(y_hats, dim=-1)

    sentence_pred = str()

    max_idx = torch.where(max_indices == 2)[0][0]

    for idx in max_indices:
        if int(idx.item()) == 2:
            break
        if idx == max_idx:
            break
        sentence_pred += str(id2char[idx.item()])
    return sentence_pred

# -----------------------------------------------------
# 서버에 상시 실행되어 있으면 예측 속도를 단축 시킬 수 있는 객체들
sentence_dict = torch.load(SENTENCE_EBD_PATH)
char2id, id2char = load_label()

model_transformer = AV_Transformer(input_vid_dim=96, num_classes=1159)
state_dict = torch.load(CKPT_PATH, map_location=device)
model_transformer.load_state_dict(state_dict["model_transformer"])
model_transformer.eval()
model_transformer = model_transformer.to(device)
# -----------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video path.')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    args = parser.parse_args()

    video_path = args.video_path
    sentence_pred = get_prediction(model_transformer, video_path)
    print(sentence_pred)