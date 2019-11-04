from torch.utils.data import Dataset
import json
import os
import cv2

class TextDataset(Dataset):
    def __init__(self, data_path, mode="train", transform=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.config = json.load(open(os.path.join(data_path, "bel_crnn.json")))
        self.transform = transform

    def abc_len(self):
        return len(self.config["abc"])

    def get_abc(self):
        return self.config["abc"]

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        #if self.mode == "test":
        #    print(f'len: {self.config[self.mode]} | {int(len(self.config[self.mode]) * 0.01)}')
            #return int(len(self.config[self.mode]) * 0.01)
        return len(self.config[self.mode])

    def __getitem__(self, idx):
        #insert cropping here nyehehehe
        name = self.config[self.mode][idx]["name"]
        text = self.config[self.mode][idx]["text"]
        xmin = self.config[self.mode][idx]["xmin"]
        ymin = self.config[self.mode][idx]["ymin"]
        xmax = self.config[self.mode][idx]["xmax"]
        ymax = self.config[self.mode][idx]["ymax"]
        
        #img_path = self.data_path + "/%s/images/%s" % (self.mode, name)
        # img = cv2.imread(os.path.join(self.data_path, "data", name))
        img = cv2.imread(name)
        img = img[ymin:ymax,xmin:xmax]
        #cv2.imwrite('img.jpg',img)
        # print(name)
        # cv2.imshow('img', cv2.resize(img, (960, 540)))
        # cv2.waitKey(0)
        seq = self.text_to_seq(text)
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": self.mode == "train"}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.config["abc"].find(c) + 1)
        return seq
