from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights
from PIL import Image
import torchvision.transforms as transforms
import argparse

from datasets import create_dataset

import sys, os

from tqdm import tqdm

import numpy as np

import csv
import pandas as pd

from scipy import linalg
import torch

def calculate_fid(feat1, feat2):
    """ Calculate FID between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch size: batch size used for inception network
    Returns:
        FID (scalar)
    """
    mu1, sigma1 = calculate_activation_statistics(feat1)
    mu2, sigma2 = calculate_activation_statistics(feat2)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid


def calculate_activation_statistics(feat):
    """Calculates the statistics used by FID
    Args:
        images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
        batch_size: batch size to use to calculate inception scores
    Returns:
        mu:     mean over all activations from the last pool layer of the inception model
        sigma:  covariance matrix over all activations from the last pool layer
                of the inception model.
    """

    feat_np = feat.cpu().detach().numpy()
    mu = np.mean(feat_np, axis=0) # (2048, 0)
    sigma = np.cov(feat_np, rowvar=False) # (2048, 2048)
    return mu, sigma


# Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


if __name__ == "__main__":
    mode = "jigsaw"

    if mode == "jigsaw":
        config = OmegaConf.load("configs/config/pretrain/jigsaw/jigsaw_custom_cityscapes.yaml")
    elif mode == "rotnet":
        config = OmegaConf.load("configs/config/pretrain/rotnet/rotnet_custom_cityscapes.yaml")
    else:
        config = OmegaConf.load("configs/config/pretrain/simclr/simclr_custom_cityscapes.yaml")
    default_config = OmegaConf.load("vissl/config/defaults.yaml")
    cfg = OmegaConf.merge(default_config, config)

    cfg = AttrDict(cfg)
    cfg.config.MODEL._MODEL_INIT_SEED = 0
    if mode == "jigsaw":
        cfg.config.MODEL.WEIGHTS_INIT.PARAMS_FILE = "checkpoints_jigsaw_cityscapes/model_phase100.torch"
    elif mode == "rotnet":
        cfg.config.MODEL.WEIGHTS_INIT.PARAMS_FILE = "checkpoints_rotnet_cityscapes/model_phase100.torch"
    else:
        cfg.config.MODEL.WEIGHTS_INIT.PARAMS_FILE = "checkpoints_simclr_cityscapes/model_phase60.torch"
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY = True
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY = True
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS = False
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP = [["res5avg", ["Identity", []]]]

    model = build_model(cfg.config.MODEL, cfg.config.OPTIMIZER)
    weights = load_checkpoint(checkpoint_path=cfg.config.MODEL.WEIGHTS_INIT.PARAMS_FILE)

    init_model_from_consolidated_weights(
        config=cfg.config,
        model=model,
        state_dict=weights,
        state_dict_key_name="classy_state_dict",
        skip_layers=[],  # Use this if you do not want to load all layers
    )

    image_size = 32
    opt = argparse.ArgumentParser()
    opt.dataroot = "dataset"
    opt.phase = "truncated_train"
    opt.gta_list = "splits/gta5/"
    opt.cityscapes_list = "splits/cityscapes/"
    opt.preprocess = "resize_and_crop"
    opt.load_size = 256
    opt.crop_size = 256
    opt.no_flip = True
    opt.batch_size = 1
    opt.dataset_mode = "cityscapes"
    opt.serial_batches = True
    opt.num_threads = 0
    opt.max_dataset_size = float("inf")

    opt.load_epoch = 0

    opt.phase = "truncated"
    opt.dataset_mode = "gta"
    # opt.dir = "datasets/gta5"
    opt.dir = "results/cycada/{}/gta5".format(opt.load_epoch)
    source_dataset = create_dataset(opt)
    opt.dir = "datasets/cityscapes"
    opt.dataset_mode = "cityscapes"
    target_dataset = create_dataset(opt)

    source_loader = source_dataset.dataloader
    target_loader = target_dataset.dataloader

    print("Loaded Epoch [{}]...".format(opt.load_epoch))

    '''
    image = Image.open("cityscapes/train/image/aachen/aachen_000000_000019_leftImg8bit.png")
    image = image.convert("RGB")

    pipeline = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    x = pipeline(image)
    '''

    model.cuda()
    feature_source = 0
    for idx, input in enumerate(tqdm(source_loader)):
        features = model(input.cuda())
        feat = torch.flatten(features[0], start_dim=1)
        if idx == 0:
            feature_source = torch.zeros((len(source_loader), feat.shape[1]))
            feature_source[idx, :] = feat
        else:
            feature_source[idx, :] = feat

    feature_target = 0
    for idx, input in enumerate(tqdm(target_loader)):
        features = model(input.cuda())
        feat = torch.flatten(features[0], start_dim=1)
        if idx == 0:
            feature_target = torch.zeros((len(target_loader), feat.shape[1]))
            feature_target[idx, :] = feat
        else:
            feature_target[idx, :] = feat

    if mode == "jigsaw":
        torch.save(feature_target, 'target_jigsaw_cityscapes.pt')
        # feature_target = torch.load('target_jigsaw_cityscapes.pt')
    elif mode == "rotnet":
        torch.save(feature_target, 'target_rotnet_cityscapes.pt')
        # feature_target = torch.load('target_rotnet_cityscapes.pt')
    else:
        # torch.save(feature_target, 'target_simclr_cityscapes.pt')
        feature_target = torch.load('target_simclr_cityscapes.pt')

    fid = calculate_fid(feature_source, feature_target)
    print("FID score", fid)

    if mode == "jigsaw":
        csv_path = os.path.join(os.getcwd(), "results", "self_supervised_results_jigsaw_cityscapes_final2.csv")
    elif mode == "rotnet":
        csv_path = os.path.join(os.getcwd(), "results", "self_supervised_results_rotnet_cityscapes_final2.csv")
    else:
        csv_path = os.path.join(os.getcwd(), "results", "self_supervised_results_simclr_cityscapes_final.csv")

    if os.path.isfile(csv_path):
        x = []
        value = []
        with open(csv_path, 'r') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(lines):
                if idx != 0:
                    x.append(row[0])
                    value.append(row[1])
        x.append(opt.load_epoch)
        value.append(fid)
        x_np = np.asarray(x).astype(int)
        value_np = np.asarray(value).astype(float)

    to_write = []
    if mode == "jigsaw":
        to_write.append(["epoch", "jigsaw"])
    elif mode == "rotnet":
        to_write.append(["epoch", "rotnet"])
    else:
        to_write.append(["epoch", "simclr"])

    if os.path.isfile(csv_path):
        for epoch in range(len(x_np)):
            result = [x_np[epoch], value_np[epoch]]
            to_write.append(result)
    else:
        result = [opt.load_epoch, fid]
        to_write.append(result)

    with open(csv_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(to_write)
