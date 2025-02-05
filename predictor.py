import torch
import numpy as np
from tqdm import tqdm

class InferenceHandler(object):
    def __init__(self, model,device,**config):
        # 只保留与推理相关的属性
        self.model = model
        self.device = device
        self.n_class = config["DECODER"]["BINARY"]

    def inference(self, dataloader):
        """
        使用已训练好的模型对新的数据进行推理（预测）。
        :param dataloader: 新数据集的 DataLoader。
        :return: 预测结果列表。
        """
        predictions = []
        att_list = []  # 用于保存每个样本的att值
        with torch.no_grad():  # 禁用梯度计算
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(tqdm(dataloader, desc="Inference Progress")):
                v_d, v_p = v_d.to(self.device), v_p.to(self.device)

                #_, _, _, score = self.model(v_d, v_p)  # 前向传播
                _, _, score, att = self.model(v_d, v_p, mode="eval")  # 使用 eval 模式获取 att 值

                if self.n_class == 1:
                    # 二分类问题
                    pred = torch.sigmoid(score).cpu().numpy()

                    # 去掉多余的维度，确保 pred 是 (batch_size,) 形状
                    pred = np.squeeze(pred, axis=1)
                else:
                    # 多分类问题
                    pred = torch.softmax(score, dim=1).cpu().numpy()

                predictions.extend(pred)
                # 将每个batch的att值保存到att_list中
                att_list.append(att.cpu().numpy())
            # 将att_list合并成一个numpy数组
        att_array = np.concatenate(att_list, axis=0)
        return predictions, att_array

    def load_model(self, model_path):
        """
        从指定路径加载预训练的模型参数。
        :param model_path: 模型参数的保存路径。
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # 加载模型参数
        self.model.to(self.device)  # 将模型移到指定设备（例如GPU或CPU）
        self.model.eval()  # 设置模型为评估模式（evaluation mode）