import torch

import torch.nn.functional as F


class AlignLoss2(torch.nn.Module):
    def __init__(self, sfer='affectnet-7', dfer='ferv39k', feat_dim=768, margin=1, decay=0.999, weight=1.0, match_method='soft'):
        super(AlignLoss2, self).__init__()
        self.id_to_class = {
            'affectnet-7': {
                0: 'Neutral',
                1: 'Happiness',
                2: 'Sadness',
                3: 'Surprise',
                4: 'Fear',
                5: 'Disgust',
                6: 'Anger',
            },
            'affectnet': {
                0: 'Neutral',
                1: 'Happiness',
                2: 'Sadness',
                3: 'Surprise',
                4: 'Fear',
                5: 'Disgust',
                6: 'Anger',
                7: 'Contempt',
            },
            'ferv39k': {
                0: 'Anger',
                1: 'Disgust',
                2: 'Fear',
                3: 'Happiness',
                4: 'Neutral',
                5: 'Sadness',
                6: 'Surprise'
            },
            'mafw': {
                0: 'Anger',
                1: 'Anxiety',
                2: 'Contempt',
                3: 'Disappointment',
                4: 'Disgust',
                5: 'Fear',
                6: 'Happiness',
                7: 'Helplessness',
                8: 'Neutral',
                9: 'Sadness',
                10: 'Surprise',
            },
            'dfew': {
                0: 'Happiness',
                1: 'Sadness',
                2: 'Neutral',
                3: 'Anger',
                4: 'Surprise',
                5: 'Disgust',
                6: 'Fear'
            }
        }
        self.margin = margin
        self.datasets = {
            'sfer': sfer,
            'dfer': dfer
        }
        # self.dfer = dfer
        self.anchors = {
            'sfer': torch.zeros(len(self.id_to_class[sfer]), feat_dim),
            'dfer': torch.zeros(len(self.id_to_class[dfer]), feat_dim)
        }
        self.decay = decay
        self.weight = weight
        self.match_method = match_method
        assert self.match_method in ['soft', 'hard']
    @torch.no_grad()
    def get_anchors(self, src, target):
        """
        将src的anchor转换为target的anchor
        """
        anchor = self.anchors[src]

        src_id_to_class = self.id_to_class[self.datasets[src]]
        target_id_to_class = self.id_to_class[self.datasets[target]]
        target_anchor = []
        for i in range(len(target_id_to_class)):
            target_class = target_id_to_class[i]
            if target_class in src_id_to_class.values():
                src_class = list(src_id_to_class.keys())[list(
                    src_id_to_class.values()).index(target_class)]
                target_anchor.append(anchor[src_class])
            else:
                # target_anchor.append(torch.zeros_like(anchor[0]))
                assert True
                # pass
        return torch.stack(target_anchor, dim=0)

    @torch.no_grad()
    def update_anchors(self, X, Y, dataset='None'):
        """
        更新anchor, momentum更新
        """
        assert dataset in ['sfer', 'dfer']
        Y = Y.argmax(dim=1)
        self.anchors[dataset] = self.anchors[dataset].to(X.device)
        for i in range(self.anchors[dataset].shape[0]):
            new_center = X[Y == i].mean(dim=0)
            if sum(new_center).isnan():
                continue
            self.anchors[dataset][i] = self.decay * self.anchors[dataset][i] \
                + (1 - self.decay) * new_center
    def forward(self, X, Y, src='sfer', target='dfer'):
        """
        anchrors: (C, D), C is the number of classes, D is the dimension of the feature
        x: (N, D), N is the number of samples
        y: (N， n_class), N is the number of samples, n_class is the number of classes
        y is mixup label
        有关MAFW数据集的还不确定
        """

        target_anchors = self.get_anchors(src, target).to(X.device)
        self.update_anchors(X, Y, target)
        A_normalized = F.normalize(target_anchors, p=2, dim=1)
        B_normalized = F.normalize(X, p=2, dim=1)

        # compute the cosine similarity
        cosine_sim = torch.mm(B_normalized, A_normalized.T)  # [N, C]
        # 计算匹配样本损失, 使用mixup收这里有两种计算方式，一种是soft考虑每个类别的相似性，一种是hard, 只考虑最大的那个类别的相似性
        #第一种soft方式：
        if self.match_method == 'soft':
            match_loss = torch.abs((Y - cosine_sim)).sum(dim=1).mean()
        else:
        # 第二种Hard方式：
            Y = Y.argmax(dim=1)
            Y = F.one_hot(Y.to(torch.int64), num_classes=len(
                self.id_to_class[self.datasets[target]]))
            match_loss = torch.abs((1 - (cosine_sim * Y)).sum(dim=1)).mean()


        if self.margin < 1:  # 涉及MAFW不能用
            # 计算不匹配样本损失
            non_match_mask = 1 - Y  # 创建一个不匹配样本的掩码
            non_match_sim = cosine_sim * non_match_mask  # 选择不匹配样本的相似度
            non_match_loss = F.relu(
                non_match_sim - self.margin).sum(dim=1).mean()  # 计算不匹配样本损失
            loss = match_loss + non_match_loss
        else:
            loss = match_loss

        return loss * self.weight


class AlignLoss(torch.nn.Module):
    def __init__(self, sfer='affectnet-7', dfer='ferv39k', feat_dim=512, margin=1, decay=0.999, weight=1.0, match_method='hard'):
        super(AlignLoss, self).__init__()
        self.id_to_class = {
            'affectnet-7': {
                0: 'Neutral',
                1: 'Happiness',
                2: 'Sadness',
                3: 'Surprise',
                4: 'Fear',
                5: 'Disgust',
                6: 'Anger',
            },
            'affectnet': {
                0: 'Neutral',
                1: 'Happiness',
                2: 'Sadness',
                3: 'Surprise',
                4: 'Fear',
                5: 'Disgust',
                6: 'Anger',
                7: 'Contempt',
            },
            'ferv39k': {
                0: 'Anger',
                1: 'Disgust',
                2: 'Fear',
                3: 'Happiness',
                4: 'Neutral',
                5: 'Sadness',
                6: 'Surprise'
            },
            'mafw': {
                0: 'Anger',
                1: 'Anxiety',
                2: 'Contempt',
                3: 'Disappointment',
                4: 'Disgust',
                5: 'Fear',
                6: 'Happiness',
                7: 'Helplessness',
                8: 'Neutral',
                9: 'Sadness',
                10: 'Surprise',
            },
            'dfew': {
                0: 'Happiness',
                1: 'Sadness',
                2: 'Neutral',
                3: 'Anger',
                4: 'Surprise',
                5: 'Disgust',
                6: 'Fear'
            }
        }
        self.margin = margin
        self.datasets = {
            'sfer': sfer,
            'dfer': dfer
        }
        # self.dfer = dfer
        self.class_embedings = torch.load('checkpoints/text_features_words.pth', map_location='cpu')

        sfer_anchors = []
        for key, value in self.id_to_class[sfer].items():
            sfer_anchors.append(self.class_embedings[value])
        dfer_anchors = []
        for key, value in self.id_to_class[dfer].items():
            dfer_anchors.append(self.class_embedings[value])
        self.anchors = {'sfer': torch.stack(sfer_anchors, dim=0), 
                        'dfer': torch.stack(dfer_anchors, dim=0)}

        self.decay = decay
        self.weight = weight
        self.match_method = match_method
        assert self.match_method in ['soft', 'hard']
    @torch.no_grad()
    def get_anchors(self, src):
        """
        将src的anchor转换为target的anchor
        """
        assert src in ['sfer', 'dfer']
        return self.anchors[src]

    @torch.no_grad()
    def update_anchors(self, X, Y, dataset='None'):
        """
        更新anchor, momentum更新
        """
        assert dataset in ['sfer', 'dfer']

    def forward(self, X, Y, src='sfer', target='dfer'):
        """
        anchrors: (C, D), C is the number of classes, D is the dimension of the feature
        x: (N, D), N is the number of samples
        y: (N， n_class), N is the number of samples, n_class is the number of classes
        y is mixup label
        有关MAFW数据集的还不确定
        """

        target_anchors = self.get_anchors(src).to(X.device)
        A_normalized = F.normalize(target_anchors, p=2, dim=1)
        B_normalized = F.normalize(X, p=2, dim=1)

        # compute the cosine similarity
        cosine_sim = torch.mm(B_normalized, A_normalized.T)  # [N, C]
        # 计算匹配样本损失, 使用mixup这里有两种计算方式，一种是soft考虑每个类别的相似性，一种是hard, 只考虑最大的那个类别的相似性
        #第一种soft方式：
        if self.match_method == 'soft':
            # match_loss = torch.abs((Y - cosine_sim)).sum(dim=1).mean()
            match_loss =  (Y - cosine_sim).sum(dim=1).mean()
        else:
        # 第二种Hard方式：
            Y = Y.argmax(dim=1)
            Y = F.one_hot(Y.to(torch.int64), num_classes=len(
                self.id_to_class[self.datasets[target]]))
            # match_loss = torch.abs((1 - (cosine_sim * Y)).sum(dim=1)).mean()
            match_loss =  - (cosine_sim * Y).sum(dim=1).mean()


        if self.margin < 1:  # 涉及MAFW不能用
            # 计算不匹配样本损失
            non_match_mask = 1 - Y  # 创建一个不匹配样本的掩码
            non_match_sim = cosine_sim * non_match_mask  # 选择不匹配样本的相似度
            non_match_loss = F.relu(
                non_match_sim - self.margin).sum(dim=1).mean()  # 计算不匹配样本损失
            loss = match_loss + non_match_loss
        else:
            loss = match_loss

        return loss * self.weight



class AlignLoss3(torch.nn.Module):
    def __init__(self, sfer='affectnet-7', dfer='ferv39k', feat_dim=512, margin=1, decay=0.999, weight=1.0, match_method='soft'):
        super(AlignLoss3, self).__init__()
        self.id_to_class = {
            'affectnet-7': {
                0: 'Neutral',
                1: 'Happiness',
                2: 'Sadness',
                3: 'Surprise',
                4: 'Fear',
                5: 'Disgust',
                6: 'Anger',
            },
            'affectnet': {
                0: 'Neutral',
                1: 'Happiness',
                2: 'Sadness',
                3: 'Surprise',
                4: 'Fear',
                5: 'Disgust',
                6: 'Anger',
                7: 'Contempt',
            },
            'ferv39k': {
                0: 'Anger',
                1: 'Disgust',
                2: 'Fear',
                3: 'Happiness',
                4: 'Neutral',
                5: 'Sadness',
                6: 'Surprise'
            },
            'mafw': {
                0: 'Anger',
                1: 'Anxiety',
                2: 'Contempt',
                3: 'Disappointment',
                4: 'Disgust',
                5: 'Fear',
                6: 'Happiness',
                7: 'Helplessness',
                8: 'Neutral',
                9: 'Sadness',
                10: 'Surprise',
            },
            'dfew': {
                0: 'Happiness',
                1: 'Sadness',
                2: 'Neutral',
                3: 'Anger',
                4: 'Surprise',
                5: 'Disgust',
                6: 'Fear'
            }
        }
        self.margin = margin
        self.datasets = {
            'sfer': sfer,
            'dfer': dfer
        }
        # self.dfer = dfer
        self.class_embedings = torch.load(
            'checkpoints/text_features_words.pth', map_location='cpu')

        sfer_anchors = []
        for key, value in self.id_to_class[sfer].items():
            sfer_anchors.append(self.class_embedings[value])
        dfer_anchors = []
        for key, value in self.id_to_class[dfer].items():
            dfer_anchors.append(self.class_embedings[value])
        self.anchors = {'sfer': torch.stack(sfer_anchors, dim=0),
                        'dfer': torch.stack(dfer_anchors, dim=0)}

        self.decay = decay
        self.weight = weight
        self.match_method = match_method
        assert self.match_method in ['soft', 'hard']

    @torch.no_grad()
    def get_anchors(self, src, v_features, y):
        """
        将src的anchor转换为target的anchor
        """
        assert src in ['sfer', 'dfer']
        t_features = self.anchors[src]
        t = []
        for i in range(len(y)):
            t.append(t_features[y[i]])
        return torch.stack(t, dim=0)

    def forward(self, X, Y, src='sfer', target='dfer'):
        """
        anchrors: (C, D), C is the number of classes, D is the dimension of the feature
        X: (N, D), N is the number of samples
        Y: (N， n_class), N is the number of samples, n_class is the number of classes
        Y is mixup label
        有关MAFW数据集的还不确定
        """
        label = torch.argmax(Y, dim=1)

        target_anchors = self.get_anchors(src, X, label).to(X.device)
        A_normalized = F.normalize(target_anchors, p=2, dim=1)
        B_normalized = F.normalize(X, p=2, dim=1)

        # compute the cosine similarity
        loss = contrastive_loss(B_normalized, A_normalized)
        return loss * self.weight

def contrastive_loss(v_features, t_features, tau=0.07):
    """
    计算对比损失
    :param v_features: normalized 视觉特征，维度为(batch_size, feature_size)
    :param t_features: normalized 文本特征，维度为(batch_size, feature_size)
    :param tau: 温度参数
    :return: 对比损失
    """
    # 计算视觉特征和文本特征之间的cosine相似度
    similarities = F.cosine_similarity(v_features.unsqueeze(1), t_features.unsqueeze(0), dim=2)
    
    # 对相似度应用指数函数并除以tau
    exp_similarities = torch.exp(similarities / tau)
    
    # 创建一个掩码，用于在相似度矩阵的对角线上选择正样本
    batch_size = v_features.size(0)
    mask = torch.eye(batch_size).bool().to(v_features.device)
    
    # 计算分子（正样本的相似度）
    numerator = exp_similarities.masked_select(mask)
    
    # 计算分母（包括所有样本的相似度）
    denominator = exp_similarities.sum(dim=1)
    
    # 对比损失
    loss = -torch.log(numerator / denominator).mean()
    
    return loss