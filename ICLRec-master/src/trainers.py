# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

from models import KMeans
from models import KMeans_Pytorch
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss, NTXent, SupConLoss, PCLoss
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr, precision_at_k


class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        self.num_intent_clusters = [int(i) for i in self.args.num_intent_clusters.split(",")]
        self.clusters = []
        self.clusters_torch = []
        for num_intent_cluster in self.num_intent_clusters:
            # initialize Kmeans
            if self.args.seq_representation_type == "mean":
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                cluster_torch = KMeans_Pytorch(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size,
                    gpu_id=self.args.gpu_id,
                    device=self.device
                )
                self.clusters.append(cluster)
                self.clusters_torch.append(cluster_torch)
            else:
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size * self.args.max_seq_length,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)
        # n_views=2, 通过阶乘等运算值=1
        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        # projection head for contrastive learn task
        self.projection = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.hidden_size, bias=True),
        )
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name；learning related："--adam_beta1"=0.9, "--adam_beta2"=0.999
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        # nelement()统计张量元素个数
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        # InfoNCELoss用于计算原文中的对比学习损失，对应原文中的L_SeqCL
        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        self.pcl_criterion = PCLoss(self.args.temperature, self.device)

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):    # 在ICLRec类中会复写这个类
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        precision, recall, ndcg = [], [], []
        for k in [5, 10, 15, 20]:
            precision.append(precision_at_k(answers, pred_list, k))
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "Precision@5": "{:.4f}".format(precision[0]),
            "Precision@10": "{:.4f}".format(precision[1]),
            "Precision@20": "{:.4f}".format(precision[3]),
            ' '
            "Recall@5": "{:.4f}".format(recall[0]),
            "Recall@10": "{:.4f}".format(recall[1]),
            "Recall@20": "{:.4f}".format(recall[3]),
            ' '
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class ICLRecTrainer(Trainer):
    # 继承Trainer类
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super(ICLRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args
        )

    def _instance_cl_one_pair_contrastive_learning(self, inputs, intent_ids=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        if self.args.seq_representation_instancecl_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        if self.args.de_noise:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids)
        else:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None)
        return cl_loss

    def _pcl_one_pair_contrastive_learning(self, inputs, intents, intent_ids):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        intents: [num_clusters batch_size hidden_dims]
        """
        n_views, (bsz, seq_len) = len(inputs), inputs[0].shape

        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model(cl_batch)

        if self.args.seq_representation_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_output_slice = torch.split(cl_sequence_flatten, bsz)
        if self.args.de_noise:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=intent_ids)
        else:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=None)
        return cl_loss

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):
        #     def train(self, epoch):
        #         self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)
        # 下面所有的dataloader==train_dataloader, cluster_dataloader==cluster_dataloader
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            # ------ 1.intentions clustering ----- #
            # 先完成交互项目的embed、负采样、聚类等任务，并不加入到反向梯度中
            if self.args.contrast_type in ["IntentCL", "Hybrid"] and epoch >= self.args.warm_up_epoches:
                print("Preparing Clustering:")
                self.model.eval()   # 如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()
                kmeans_training_data = []
                rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))

                # 下面循环得到的kmeans_training_data：list(140)，每组是batch_size个用户交互项目的embedding
                for i, (rec_batch, _, _) in rec_cf_data_iter:       # 调用 RecWithContrastiveLearningDataset类的__getitem__函数，返回（）里的三个tensor
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)     # rec_batch是进行了负采样的交互数据，没有数据增强。一个batch，最终维度是[256, 50]
                    _, input_ids, target_pos, target_neg, _ = rec_batch
                    # 运行SASRecModel模型的forward函数，完成工作：
                    # 1.加入mask
                    # 2.对项目编码+位置embedding
                    # 3.调用自监督tup
                    # 4.调用intermediate
                    # 5.返回最终的一个(256, 50, 64)的向量
                    sequence_output = self.model(input_ids)
                    # average sum
                    if self.args.seq_representation_type == "mean":
                        sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)     # (256, 64)得到每个用户交互项目的特征（将50个平均为1个），理解为项目画像？
                    sequence_output = sequence_output.view(sequence_output.shape[0], -1)    # view()调整tensor的形状
                    sequence_output = sequence_output.detach().cpu().numpy()    # detach()将variable参数从网络中隔离开，不参与参数更新
                    kmeans_training_data.append(sequence_output)
                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0) # (35598, 64)

                # train multiple clusters
                print("Training Clusters:")
                a = self.clusters
                b = len(self.clusters)
                for i, cluster in tqdm(enumerate(self.clusters), total=len(self.clusters)):
                    centroid = cluster.train(kmeans_training_data)     # 调用k-means模型，返回K-mean对象
                    self.clusters[i] = cluster
                print('')
                # clean memory
                del kmeans_training_data    # 最后又删掉这个对象？？？
                import gc

                gc.collect()    # 垃圾回收

            # ------ 2.model training -----#
            print("Performing Rec model Training:")
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:

                """
                调用RecWithContrastiveLearningDataset类的__getitem__
                rec_batch-->cur_rec_tensors:  含有5个tensor的合集（正样本、负样本、input等）
                cl_batches-->cf_tensors_list:  含有2个tensor的合集（经过2种数据增强返回的）
                seq_class_label_batches-->seq_class_label:  序列监督信号
                
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                """
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                sequence_output = self.model(input_ids)     # 调用SASRec模型，返回最终的一个(256, 50, 64)
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)  # 返回的是一个loss值

                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    # cl_batches：含有2个tensor的合集（经过2种数据增强返回的）
                    # if self.args.contrast_type == "InstanceCL":
                    #     cl_loss = self._instance_cl_one_pair_contrastive_learning(
                    #         cl_batch, intent_ids=seq_class_label_batches
                    #     )
                    #     cl_losses.append(self.args.cf_weight * cl_loss)
                    # elif self.args.contrast_type == "IntentCL":
                    #     # ------ performing clustering for getting users' intentions ----#
                    #     # average sum
                    #     if epoch >= self.args.warm_up_epoches:
                    #         if self.args.seq_representation_type == "mean":
                    #             sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                    #         sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                    #         sequence_output = sequence_output.detach().cpu().numpy()
                    #
                    #         # query on multiple clusters
                    #         for cluster in self.clusters:
                    #             seq2intents = []
                    #             intent_ids = []
                    #             intent_id, seq2intent = cluster.query(sequence_output)
                    #             seq2intents.append(seq2intent)
                    #             intent_ids.append(intent_id)
                    #         cl_loss = self._pcl_one_pair_contrastive_learning(
                    #             cl_batch, intents=seq2intents, intent_ids=intent_ids
                    #         )
                    #         cl_losses.append(self.args.intent_cf_weight * cl_loss)
                    #     else:
                    #         continue
                    if self.args.contrast_type == "Hybrid":
                        if epoch < self.args.warm_up_epoches:
                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )   # 返回一个对增强数计算的loss
                            cl_losses.append(self.args.cf_weight * cl_loss1)
                        else:
                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)    # cf_weight默认0.1
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = sequence_output.detach().cpu().numpy()    # 再次进行平均（256， 64）
                            # query on multiple clusters
                            for cluster in self.clusters:
                                seq2intents = []
                                intent_ids = []
                                intent_id, seq2intent = cluster.query(sequence_output)  # 调用K-means函数的query
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                            cl_loss3 = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids
                            )
                            cl_losses.append(self.args.intent_cf_weight * cl_loss3) # intent_cf_weight默认0.3

                joint_loss = self.args.rec_weight * rec_loss    # rec_weight默认1.0
                for cl_loss in cl_losses:
                    # 把原始的rec_loss和对比学习的loss相加
                    joint_loss += cl_loss      #   cl_loss集合里面有self.args.cf_weight * cl_loss1；self.args.intent_cf_weight * cl_loss3
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()

                for i, cl_loss in enumerate(cl_losses):
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            # train = False
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]    # 输出top20的索引值[256, 20]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]    # [256, 20]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]    # [256, 20]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]    # [256, 20]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                # answer_list[35598, 1]=[n_users, 1];
                # pred_list[35598, 20]
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
