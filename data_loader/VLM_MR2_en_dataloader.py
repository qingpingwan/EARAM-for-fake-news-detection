import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader



torch.manual_seed(666)

def read_txt_to_list(file_path):
    """ 
    读取指定路径的 TXT 文档,并将每一行作为一个字符串存储在列表中。
    
    Args:
        file_path (str): TXT 文档的路径。
        
    Returns:
        list: 包含 TXT 文档每一行的字符串列表。
    """
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: {file_path} does not exist.")
        return []
    except IOError:
        print(f"Error: Unable to read {file_path}.")
        return []
    return lines


analy_1_en_train = read_txt_to_list("./data_3_MR2/MR2_en_train_analysis_1.txt")

analy_2_en_train = read_txt_to_list("./data_3_MR2/MR2_en_train_analysis_2_RAG.txt")

analy_1_en_test = read_txt_to_list("./data_3_MR2/MR2_en_test_analysis_1.txt")

analy_2_en_test = read_txt_to_list("./data_3_MR2/MR2_en_test_analysis_2_RAG.txt")



class ImageCaptionDataset(Dataset):
    def __init__(self, json_file, img_dir, analy_1, analy_2):
        """
        初始化 ImageCaptionDataset 类
        
        参数:
        json_file (str): 输入 JSON 文件的路径
        img_dir (str): 图像文件所在的目录
        """
        self.data = self.load_data(json_file)
        self.img_dir = img_dir
        self.analy_1 = analy_1
        self.analy_2 = analy_2

        
    def load_data(self, json_file):
        """
        从 JSON 文件中加载数据
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        根据索引返回caption文本、图像和标签
        """
        key = list(self.data.keys())[idx]
        item = self.data[key]
        
        caption = item['caption']
        img_path = os.path.join(self.img_dir, item['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = item['label']
        
        return caption, image, self.analy_1[idx], self.analy_2[idx], torch.tensor(label)
    
    
def custom_collate(batch):
    texts, images, data1, data2, labels = zip(*batch)
    
    # 保持 images 和 texts 为列表格式
    images = list(images)
    texts = list(texts)
    data1 = list(data1)
    data2 = list(data2)
    # 将 labels 堆叠成张量
    labels = torch.stack(labels, dim=0)
    
    return  texts, images, data1, data2, labels


# Usage example
def load_test_MR2(batch_size):
    all_dataset = ImageCaptionDataset('./data_3_MR2/dataset_merge/en_val.json', './data_3_MR2/dataset_merge', analy_1_en_test, analy_2_en_test)

    # Create the data loaders
    
    test_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return test_loader

def load_train_MR2(batch_size):
    all_dataset = ImageCaptionDataset('./data_3_MR2/dataset_merge/en_train.json', './data_3_MR2/dataset_merge',analy_1_en_train,analy_2_en_train)

    # Create the data loaders
    
    test_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return test_loader


if __name__ == "__main__":
    dataload = load_test_MR2(1)
    for x1,x2,x3,x4,x5 in dataload:
        print(x3)
        break
    