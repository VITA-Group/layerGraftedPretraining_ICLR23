import os
import numpy as np

from dataset.customDataset import Custom_Dataset

def init_datasets(args, transform_train, transform_test):
    if args.dataset == "imagenet":
        # Data loading code
        root, txt_train, txt_val, txt_test, pathReplaceDict = get_imagenet_root_split(args.data, args.customSplit)

        train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train, pathReplace=pathReplaceDict)
        val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test, pathReplace=pathReplaceDict)
        if hasattr(args, "CAM") and args.CAM:
            print("imagenet_val_shuffle")
            print("imagenet_val_shuffle")
            print("imagenet_val_shuffle")
            print("imagenet_val_shuffle")
            test_datasets = Custom_Dataset(root=root, txt="split/imagenet/imagenet_val_shuffle.txt", transform=transform_test, pathReplace=pathReplaceDict)
        else:
            test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test, pathReplace=pathReplaceDict)
    elif args.dataset == "imagenet100":
        # Data loading code
        root, txt_train, txt_val, txt_test, pathReplaceDict = get_imagenet100_root_split(args.data, args.customSplit)

        train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train, pathReplace=pathReplaceDict)
        val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test, pathReplace=pathReplaceDict)
        test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test, pathReplace=pathReplaceDict)
    elif args.dataset == 'Pet37':
        root, txt_train, txt_val, txt_test = get_pet37_data_split(args.data, args.customSplit)

        train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
        val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
        test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)
    elif args.dataset == 'food101':
        root, txt_train, txt_val, txt_test = get_food101_data_split(args.data, args.customSplit)

        train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
        val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
        test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)
    else:
        raise ValueError("No such dataset: {}".format(args.dataset))

    return train_datasets, val_datasets, test_datasets


def get_imagenet_root_path(root):

    pathReplaceDict = {}
    if os.path.isdir(root):
        if os.path.isdir(os.path.join(root, "train_new")):
            pathReplaceDict = {"train/": "train_new/"}
    elif os.path.isdir("/ssd1/bansa01/imagenet_final"):
        root = "/ssd1/bansa01/imagenet_final"
    elif os.path.isdir("/mnt/imagenet"):
        root = "/mnt/imagenet"
    elif os.path.isdir("/ssd1/yifan/imagenet/"):
        root = "/ssd1/yifan/imagenet/"
    elif os.path.isdir("/hdd3/ziyu/imagenet"):
        root = "/hdd3/ziyu/imagenet"
    elif os.path.isdir("/ssd2/invert/imagenet_final/"):
        root = "/ssd2/invert/imagenet_final/"
    elif os.path.isdir("/home/yucheng/imagenet"):
        root = "/home/yucheng/imagenet"
    elif os.path.isdir("/data/datasets/ImageNet_unzip"):
        root = "/data/datasets/ImageNet_unzip"
    elif os.path.isdir("/scratch/user/jiangziyu/imageNet"):
        root = "/scratch/user/jiangziyu/imageNet"
    elif os.path.isdir("/datadrive_d/imagenet"):
        root = "/datadrive_d/imagenet"
    elif os.path.isdir("/datadrive_c/imagenet"):
        root = "/datadrive_c/imagenet"
    elif os.path.isdir("/ssd2/tianlong/zzy/data/imagenet"):
        root = "/ssd2/tianlong/zzy/data/imagenet"
    elif os.path.isdir("/home/xinyu/dataset/imagenet2012"):
        root = "/home/xinyu/dataset/imagenet2012"
    elif os.path.isdir("/home/xueq13/scratch/ziyu/ImageNet/ILSVRC/Data/CLS-LOC"):
        root = "/home/xueq13/scratch/ziyu/ImageNet/ILSVRC/Data/CLS-LOC"
    elif os.path.isdir("/hdd1/ziyu/ImageNet/"):
        root = "/hdd1/ziyu/ImageNet/"
    elif os.path.isdir("/mnt/models/imagenet_new"):
        root = "/mnt/models/imagenet_new"
        pathReplaceDict = {"train/": "train_new/"}
    elif os.path.isdir("/mnt/models/imagenet"):
        root = "/mnt/models/imagenet"
    elif os.path.isdir("/home/ziyu/dataset"):
        root = "/home/ziyu/dataset"
    elif os.path.isdir("/home/grads/x/xueq13/dataset/imagenet"):
        root = "/home/grads/x/xueq13/dataset/imagenet"
    elif os.path.isdir("/ssd1/ziyu/imagenet"):
        root = "/ssd1/ziyu/imagenet"
    else:
        print("No dir for imagenet")
        assert False

    return root, pathReplaceDict


def get_imagenet_root_split(root, customSplit, domesticAnimalSplit=False):
    root, pathReplaceDict = get_imagenet_root_path(root)

    txt_train = "split/imagenet/imagenet_train.txt"
    txt_val = "split/imagenet/imagenet_val.txt"
    txt_test = "split/imagenet/imagenet_val.txt"

    if domesticAnimalSplit:
        txt_train = "split/imagenet/imagenet_domestic_train.txt"
        txt_val = "split/imagenet/imagenet_domestic_val.txt"
        txt_test = "split/imagenet/imagenet_domestic_test.txt"

    if customSplit != '':
        txt_train = "split/imagenet/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test, pathReplaceDict


def get_imagenet100_root_split(root, customSplit):
    root, pathReplaceDict = get_imagenet_root_path(root)

    txt_train = "split/imagenet/ImageNet_100_train.txt"
    txt_val = "split/imagenet/ImageNet_100_val.txt"
    txt_test = "split/imagenet/ImageNet_100_test.txt"

    if customSplit != '':
        txt_train = "split/imagenet/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test, pathReplaceDict


def getPlacesRoot(root):
    if os.path.isdir(root):
        pass
    if os.path.isdir("/hdd2/ziyu/places365"):
        root = "/hdd2/ziyu/places365"
    elif os.path.isdir("/scratch/user/jiangziyu/places365"):
        root = "/scratch/user/jiangziyu/places365"
    elif os.path.isdir("/home/xueq13/scratch/ziyu/Places"):
        root = "/home/xueq13/scratch/ziyu/Places"
    else:
        raise NotImplementedError("no root")

    return root



def get_cifar10_data_split(root, customSplit, ssl=False):
    # if ssl is True, use both train and val splits
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir('../../data'):
            root = '../../data'
        elif os.path.isdir('/mnt/models/dataset/'):
            root = '/mnt/models/dataset/'
        elif os.path.isdir('/mnt/models/Ziyu'):
            root = '/mnt/models/Ziyu'
        else:
            assert False

    if ssl:
        assert customSplit == ''
        train_idx = "split/cifar10/trainValIdxList.npy"
        return root, train_idx, None

    train_idx = "split/cifar10/trainIdxList.npy"
    val_idx = "split/cifar10/valIdxList.npy"
    if customSplit != '':
        train_idx = "split/cifar10/{}.npy".format(customSplit)

    return root, train_idx, val_idx


def get_STL10_data_split(root, customSplit, ssl=False):
    # if ssl is True, use both train and val splits
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir('/hdd1/ziyu/MSR/data/stl10_binary'):
            root = '/hdd1/ziyu/MSR/data/stl10_binary'
        else:
            assert False

    if ssl:
        assert customSplit == ''
        train_idx = "split/STL10/trainValIdxList.npy"
        return root, train_idx, None

    train_idx = "split/STL10/trainIdxList.npy"
    val_idx = "split/STL10/valIdxList.npy"
    if customSplit != '':
        train_idx = "split/STL10/{}.npy".format(customSplit)

    return root, train_idx, val_idx


def get_cifar100_data_split(root, customSplit, ssl=False):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir('../../data'):
            root = '../../data'
        elif os.path.isdir('/mnt/models/dataset/'):
            root = '/mnt/models/dataset/'
        elif os.path.isdir('/mnt/models/'):
            root = '/mnt/models/'
        else:
            root = '.'

    if ssl:
        assert customSplit == ''
        train_idx = "split/cifar100/cifar100_trainValIdxList.npy"
        return root, train_idx, None

    train_idx = "split/cifar100/cifar100_trainIdxList.npy"
    val_idx = "split/cifar100/cifar100_valIdxList.npy"
    if customSplit != '':
        train_idx = "split/cifar100/{}.npy".format(customSplit)

    return root, train_idx, val_idx

def get_iNaturalist_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/datadrive_c/ziyu/iNaturalist/"):
            root = "/datadrive_c/ziyu/iNaturalist/"
        elif os.path.isdir("/mnt/inaturalist/"):
            root = "/mnt/inaturalist/"
        elif os.path.isdir("/home/xueq13/scratch/ziyu/iNaturalist"):
            root = "/home/xueq13/scratch/ziyu/iNaturalist"
        elif os.path.isdir("/hdd1/ziyu/MSR/data"):
            root = "/hdd1/ziyu/MSR/data"
        elif os.path.isdir('/mnt/models/'):
            root = '/mnt/models/'
        else:
            assert False

    return root


def get_iNaturalist_data_split(root, customSplit):
    root = get_iNaturalist_path(root)

    txt_train = "split/iNaturalist/iNaturalist18_train.txt"
    txt_val = "split/iNaturalist/iNaturalist18_val.txt"
    txt_test = "split/iNaturalist/iNaturalist18_val.txt"

    if customSplit != '':
        txt_train = "split/iNaturalist/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_iNaturalist_sub1000_data_split(root, customSplit):
    root = get_iNaturalist_path(root)

    txt_train = "split/iNaturalist/iNaturalist18_sub1000_train.txt"
    txt_val = "split/iNaturalist/iNaturalist18_sub1000_val.txt"
    txt_test = "split/iNaturalist/iNaturalist18_sub1000_val.txt"

    if customSplit != '':
        txt_train = "split/iNaturalist/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_places_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/home/t-ziyujiang/dataset"):
            root = "/home/t-ziyujiang/dataset"
        elif os.path.isdir("/mnt/places365"):
            root = "/mnt/places365"
        else:
            assert False

    return root


def get_pet37_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/hdd1/ziyu/MSR/data/Pet37/images"):
            root = "/hdd1/ziyu/MSR/data/Pet37/images"
        elif os.path.isdir("/home/xueq13/scratch/ziyu/downstream_data/Pet37/images"):
            root = "/home/xueq13/scratch/ziyu/downstream_data/Pet37/images"
        elif os.path.isdir('/mnt/models/Pet37/images/'):
            root = '/mnt/models/Pet37/images/'
        elif os.path.isdir('/ssd1/ziyu/data/Pet37/images/'):
            root = '/ssd1/ziyu/data/Pet37/images/'
        else:
            assert False

    return root


def get_pet37_data_split(root, customSplit, ssl=False):
    root = get_pet37_path(root)

    txt_train = "split/Pet37/Pet37_train.txt"
    txt_val = "split/Pet37/Pet37_val.txt"
    txt_test = "split/Pet37/Pet37_test.txt"

    if customSplit != '':
        txt_train = "split/Pet37/{}.txt".format(customSplit)

    if ssl:
        assert customSplit == ''
        train_idx = "split/Pet37/Pet37_trainval.txt"
        return root, train_idx, None, None

    return root, txt_train, txt_val, txt_test


def get_food101_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/hdd1/ziyu/MSR/data/food-101/images"):
            root = "/hdd1/ziyu/MSR/data/food-101/images"
        elif os.path.isdir('/mnt/models/food-101/images/'):
            root = '/mnt/models/food-101/images/'
        else:
            assert False

    return root


def get_food101_data_split(root, customSplit, ssl=False):
    root = get_food101_path(root)

    txt_train = "split/food-101/food101_train.txt"
    txt_val = "split/food-101/food101_val.txt"
    txt_test = "split/food-101/food101_test.txt"

    if customSplit != '':
        txt_train = "split/food-101/{}.txt".format(customSplit)

    if ssl:
        assert customSplit == ''
        train_idx = "split/food-101/food101_trainval.txt"
        return root, train_idx, None, None

    return root, txt_train, txt_val, txt_test


def get_dtd_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/hdd1/ziyu/MSR/data/dtd/images"):
            root = "/hdd1/ziyu/MSR/data/dtd/images"
        elif os.path.isdir('/mnt/models/dtd/images/'):
            root = '/mnt/models/dtd/images/'
        else:
            assert False

    return root


def get_dtd_data_split(root, customSplit, ssl=False):
    root = get_dtd_path(root)

    txt_train = "split/dtd/dtd_train.txt"
    txt_val = "split/dtd/dtd_val.txt"
    txt_test = "split/dtd/dtd_test.txt"

    if customSplit != '':
        txt_train = "split/dtd/{}.txt".format(customSplit)

    if ssl:
        assert customSplit == ''
        train_idx = "split/dtd/dtd_trainval.txt"
        return root, train_idx, None, None

    return root, txt_train, txt_val, txt_test


def get_EuroSAT_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/hdd1/ziyu/MSR/data/EuroSAT/2750"):
            root = "/hdd1/ziyu/MSR/data/EuroSAT/2750"
        else:
            assert False

    return root


def get_EuroSAT_data_split(root, customSplit, ssl=False):
    root = get_EuroSAT_path(root)

    txt_train = "split/EuroSAT/EuroSAT_train.txt"
    txt_val = "split/EuroSAT/EuroSAT_val.txt"
    txt_test = "split/EuroSAT/EuroSAT_test.txt"

    if customSplit != '':
        txt_train = "split/EuroSAT/{}.txt".format(customSplit)

    if ssl:
        assert customSplit == ''
        train_idx = "split/EuroSAT/EuroSAT_trainval.txt"
        return root, train_idx, None, None

    return root, txt_train, txt_val, txt_test


def get_places_sub100_data_split(root, customSplit):
    root = get_places_path(root)

    txt_train = "split/places365/places365_sub100_train.txt"
    txt_val = "split/places365/places365_sub100_val.txt"
    txt_test = "split/places365/places365_sub100_val.txt"

    if customSplit != '':
        txt_train = "split/places365/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_chexpert_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/home/xueq13/scratch/ziyu/chestXpert/CheXpert-v1.0-small/CheXpert-v1.0-small"):
            root = "/home/xueq13/scratch/ziyu/chestXpert/CheXpert-v1.0-small/CheXpert-v1.0-small"
        elif os.path.isdir("../../data/CheXpert-v1.0-small"):
            root = "../../data/CheXpert-v1.0-small"
        else:
            assert False

    return root


def get_chexpert_data_split(root, customSplit):
    root = get_chexpert_path(root)

    txt_train = "split/chexpert/train.0-50k.csv"
    txt_val = "split/chexpert/train.50k-100k.csv"
    txt_test = "split/chexpert/train.50k-100k.csv"

    if customSplit != '':
        txt_train = "split/chexpert/{}.csv".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_aircraft_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("../../data/fgvc_aircraft"):
            root = "../../data/fgvc_aircraft"
        else:
            assert False

    return root


def get_aircraft_data_split(root, customSplit):
    root = get_aircraft_path(root)

    txt_train = "split/aircraft/train_100.txt"
    txt_val = "split/aircraft/test.txt"
    txt_test = "split/aircraft/test.txt"

    if customSplit != '':
        txt_train = "split/aircraft/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_Caltech101_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/hdd1/ziyu/MSR/data/101_ObjectCategories"):
            root = "/hdd1/ziyu/MSR/data/101_ObjectCategories"
        elif os.path.isdir('/mnt/models/101_ObjectCategories/'):
            root = '/mnt/models/101_ObjectCategories/'
        else:
            assert False

    return root


def get_Caltech101_data_split(root, customSplit, ssl=False):
    root = get_Caltech101_path(root)

    txt_train = "split/Caltech101/Caltech101_train.txt"
    txt_val = "split/Caltech101/Caltech101_val.txt"
    txt_test = "split/Caltech101/Caltech101_test.txt"

    if customSplit != '':
        txt_train = "split/Caltech101/{}.txt".format(customSplit)

    if ssl:
        assert customSplit == ''
        train_idx = "split/Caltech101/Caltech101_trainval.txt"
        return root, train_idx, None, None

    return root, txt_train, txt_val, txt_test


def get_SUN397_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/hdd1/ziyu/MSR/data/SUN397"):
            root = "/hdd1/ziyu/MSR/data/SUN397"
        elif os.path.isdir('/mnt/models/SUN397'):
            root = '/mnt/models/SUN397'
        else:
            assert False

    return root


def get_SUN397_data_split(root, customSplit, ssl=False):
    root = get_SUN397_path(root)

    txt_train = "split/SUN397/SUN397_train.txt"
    txt_val = "split/SUN397/SUN397_val.txt"
    txt_test = "split/SUN397/SUN397_test.txt"

    if customSplit != '':
        txt_train = "split/SUN397/{}.txt".format(customSplit)

    if ssl:
        assert customSplit == ''
        train_idx = "split/SUN397/SUN397_trainval.txt"
        return root, train_idx, None, None

    return root, txt_train, txt_val, txt_test
