# --- IMPORTS
import torchvision, jellyfish, warnings, scipy.io, zipfile, gensim, urllib, torch, numpy, json, math, clip, os
from scipy.stats import normaltest, shapiro, mannwhitneyu
from PIL import ImageFont, ImageDraw, Image
import matplotlib.patches as mpatches
from matplotlib import font_manager
import matplotlib.pyplot as plt
import torch.nn.functional as F
import statistics

# SCRIPTS
import FONTS.get_fonts
import UTILS.set_up_matplotlib
import DATASETS.get_dataset
from DATASETS.load_dataset import get_images
import DATASETS.LABELS.load_labels
from evaluation import cal_evaluation

if __name__ == '__main__':
    images = get_images()
    # LABELS
    labels = torch.load("DATASETS/LABELS/labels.pt")
    super_labels, basic_labels, hierarchy = labels["SUPERORDINATES"], labels["BASICS"], labels["CLUSTERING"]

    # MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model_name = "clip"  # Name of the model
    preprocess_fn = preprocess  # Preprocessing to be applied on raw images
    tokenize_fn = clip.tokenize  # Tokenize function

    contexts = ["a photo of a "]
    dataset_size = 274
    model_w2v = gensim.models.KeyedVectors.load_word2vec_format("DATASETS/GoogleNews-vectors-negative300.bin", binary=True)

    cal_evaluation() #the results are saved with the model name in the data directory.
