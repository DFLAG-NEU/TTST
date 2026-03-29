import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,Dataset
import random
import numpy as np
import os
import pandas as pd
from torchvision.datasets.folder import pil_loader




logger = logging.getLogger(__name__)


# class my_ImageFolder(datasets.ImageFolder):
#     def __init__(self,root, transform):
#         super(my_ImageFolder, self).__init__(root, transform)
#         feature_path = 'normalized_features'
#         self.df_all = pd.DataFrame(data=None)
#         for r, d, files in os.walk(feature_path):
#             for f in files:
#                 file_path = os.path.join(r, f)
#                 df = pd.read_excel(file_path)
#                 col = df.columns[2:]
#                 df = df[col]
#                 self.df_all = pd.concat([self.df_all, df], axis=1)
#
#
#
#
#
#     def __getitem__(self, index: int):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, target = self.samples[index]
#         mask_path = path.selected_img.replace('1_','').replace('2_','').replace('3_','').replace('4_','').replace('5_','').replace('6_','').replace('7_','').replace('8_','').replace('train', 'croped_mask_png').replace('test', 'croped_mask_png')
#         sample = self.loader(path)
#         mask = self.loader(mask_path)
#         if self.transform is not None:
#             seed = np.random.randint(2147483647)
#             random.seed(seed)  # apply this seed to img tranfsorms
#             sample = self.transform(sample)
#             random.seed(seed)  # apply this seed to img tranfsorms
#             mask = self.transform(mask)
#
#             # sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         information = path.split('\\')
#         radio_feature_name = information[-4]+'____'+information[-2]+'____'+information[-1].replace('.png','')
#         feature = self.df_all[radio_feature_name]
#         feature = feature.to_numpy()
#         feature = torch.from_numpy(feature)
#         return sample, mask, feature, target

class my_ImageFolder(datasets.ImageFolder):
    def __init__(self,root, transform):
        super(my_ImageFolder, self).__init__(root, transform)
        feature_path = 'normalized_features_CSV'
        self.df_all = pd.DataFrame(data=None)
        for r, d, files in os.walk(feature_path):
            for f in files:
                file_path = os.path.join(r, f)
                df = pd.read_csv(file_path)
                col = df.columns[2:]
                df = df[col]
                self.df_all = pd.concat([self.df_all, df], axis=1)





    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # print('***********',index,path, target)
        mask_path = path.replace('1_', '').replace('2_', '').replace('3_', '').replace('4_', '').replace('5_', '').replace('6_', '').replace('7_', '').replace('8_', '').replace('train', 'croped_mask_png').replace('test', 'croped_mask_png')
        sample = self.loader(path)
        mask = self.loader(mask_path)
        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)  # apply this seed to img tranfsorms
            sample = self.transform(sample)
            random.seed(seed)  # apply this seed to img tranfsorms
            mask = self.transform(mask)

            # sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, mask,  target




class sequence_dataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.root = root
        class_list = os.listdir(root)
        class_dic = {}
        for cls in class_list:
            class_dic[cls] = {}
            patient_list = os.listdir(os.path.join(root, cls))
            for patient in patient_list:
                class_dic[cls][patient] = {}
                location_list = os.listdir(os.path.join(root, cls, patient, 'SA'))
                for location in location_list:
                    class_dic[cls][patient][location] = []
                    img_list = os.listdir(os.path.join(root, cls, patient, 'SA', location))
                    for img in img_list:
                        img_path = os.path.join(root, cls, patient, 'SA', location, img)
                        class_dic[cls][patient][location].append(img_path)
        self.class_dic = class_dic
        patient_num = 0
        for c in class_dic.values():
            patient_num += len(c)
        self.patient_num = patient_num
        self.getitem_dic = {}
        self.getitem_index_key = {}
        self.target_dic = {}
        i = 0
        for cls, cls_v in class_dic.items():
            for patient, patient_v in cls_v.items():
                self.getitem_dic[cls + '_' + patient] = patient_v
                self.target_dic[cls + '_' + patient] = cls
                self.getitem_index_key[i] = cls + '_' + patient
                i = i + 1





    def __getitem__(self, index: int):
        patient_name = self.getitem_index_key[index]
        patient = self.getitem_dic[self.getitem_index_key[index]]
        target_type =  self.target_dic[self.getitem_index_key[index]]
        if target_type  == 'mace_cine':
            target = 0
        else:
            target = 1
        sample_list = []
        mask_list = []
        for loc, img_list in patient.items():
            # selected_img = random.choice(img_list)
            selected_img = img_list[0:25]
            sample_list.extend(selected_img)
            selected_mask = img_list[0:25]
            for index, name in enumerate(selected_mask):
                selected_mask[index] = name.replace('1_','').replace('2_','').replace('3_','').replace('4_','').replace('train', 'croped_mask_png').replace('test', 'croped_mask_png').replace('total', 'croped_mask_png')
            mask_list.extend(selected_mask)
        # print(self.getitem_index_key[index])
        LGE_path = os.path.join(os.path.dirname(self.root), "ARVC_LGE_PNG_Square_croped", target_type, patient_name.replace(target_type + "_", ''))
        # LGE_mask_path = os.path.join(os.path.dirname(self.root), "ARVC_LGE_PNG_Square_Mask_croped", target_type, patient_name.replace(target_type+"_", ''))
        if os.path.exists(LGE_path):
            LGE_image_list = []
            LGE_mask_list = []
            for i in os.listdir(LGE_path):
                LGE_image = pil_loader(os.path.join(LGE_path, i))
                LGE_mask = pil_loader(os.path.join(LGE_path, i).replace("ARVC_LGE_PNG_Square_croped", "ARVC_LGE_PNG_Square_Mask_croped"))
                if self.transform is not None:
                    seed = np.random.randint(2147483647)
                    # random.seed(seed)  # apply this seed to img tranfsorms
                    torch.manual_seed(seed)
                    LGE_image = self.transform(LGE_image)
                    # random.seed(seed)  # apply this seed to img tranfsorms
                    torch.manual_seed(seed)
                    LGE_mask = self.transform(LGE_mask)
                LGE_image_list .append(LGE_image)
                LGE_mask_list.append(LGE_mask)

            if len(LGE_image_list) < 12:
                original_seq_length = len(LGE_image_list)
                i=original_seq_length
                while i < 12:
                    LGE_image_list.append(LGE_image_list[i%original_seq_length])
                    LGE_mask_list.append(LGE_mask_list[i%original_seq_length])
                    i += 1

            if len(LGE_image_list) > 12:
                LGE_image_list = LGE_image_list[:12]
                LGE_mask_list = LGE_mask_list[:12]
            LGE_images = torch.stack(LGE_image_list, dim=0)
            LGE_masks = torch.stack(LGE_mask_list, dim=0)
        else:
            LGE_images = torch.zeros([12,3,224,224])
            LGE_masks = torch.zeros([12,3,224,224])




        sample_img_list = []
        mask_img_list = []

        for i in range(len(sample_list)):
            im = pil_loader(sample_list[i])
            mk = pil_loader(mask_list[i])
            if self.transform is not None:
                seed = np.random.randint(2147483647)
                random.seed(seed)  # apply this seed to img tranfsorms
                im = self.transform(im)
                random.seed(seed)  # apply this seed to img tranfsorms
                mk = self.transform(mk)
            sample_img_list.append(im)
            mask_img_list.append(mk)


        if len(sample_img_list)<300:
            original_seq_length = len(sample_img_list)
            i=original_seq_length
            while i < 300:
                sample_img_list.append(sample_img_list[i%original_seq_length])
                mask_img_list.append(mask_img_list[i%original_seq_length])

                i += 1
        if len(sample_img_list) > 300:
            sample_img_list = sample_img_list[:300]
            mask_img_list = mask_img_list [:300]

        sample = torch.stack(sample_img_list, dim=0)
        mask = torch.stack(mask_img_list, dim=0)




        return sample, mask, LGE_images, LGE_masks, target

    def __len__(self):
        return self.patient_num


def get_loader(args):
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    transform_train = transforms.Compose([
        # color_jitter,
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.9, 1.0)),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, expand=False),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])



    trainset = sequence_dataset(root=args.train_data_folder,
                                     transform=transform_train)
    testset = sequence_dataset(root=args.test_data_folder,
                                      transform=transform_test)



    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=0,
                              pin_memory=False)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=False) if testset is not None else None

    return train_loader, test_loader

def get_test_loader(args):
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)


    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])




    testset = sequence_dataset(root=args.test_data_folder,
                                      transform=transform_test)




    test_sampler = SequentialSampler(testset)

    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=False) if testset is not None else None

    return test_loader


def get_loader_img(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        # transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(45, expand=False),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None
    elif args.dataset == "MACE":

        trainset = my_ImageFolder(root=args.train_data_folder,
                                         transform=transform_train)
        testset = my_ImageFolder(root=args.test_data_folder,
                                         transform=transform_test)
    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=False)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=False) if testset is not None else None

    return train_loader, test_loader
