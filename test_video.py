# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from lib import segmentation


def main(args):
    model = segmentation.__dict__[args.model](num_classes=2,
        aux_loss=False,
        pretrained=False,
        args=args)
    model.eval()

    # 加載語言模型（例如，BERT或GPT-3）和其標記器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    language_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

    # 定義影片讀取器
    video_path = 'test.mp4'
    cap = cv2.VideoCapture(video_path)
    # 定義轉換器，用於預處理影格
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    # 定義輸出影片寫入器
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (568, 320), isColor=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_frame = transform(frame)
        with torch.no_grad():
            mask = model(input_frame.unsqueeze(0).cuda())
        mask = mask.squeeze(0).cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        masked_description = "REMOVE the baby."
        input_ids = tokenizer.encode(masked_description, return_tensors="pt")
        with torch.no_grad():
            output = language_model.generate(input_ids, max_length=50, num_return_sequences=1)
        description = tokenizer.decode(output[0], skip_special_tokens=True)
        cv2.putText(frame, description, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)