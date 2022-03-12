# Language-biased image classification: evaluation based on semantic representations
This repository shares the source codes used in our ICLR 2022 paper, "Language-biased image classification:
evaluation based on semantic representations" by Lemesle*, Y., Sawayama*, M., Valle-Perez, G., Adolphe, M., Sauzéon, H., & Oudeyer, P. Y.
(*equal contribution)


The source codes are to create the word-superimposed images
from the image datasets (Cichy et al., 2016; Mohsenzadeh et al., 2019) 
and to reproduce the analysis results shown in the paper. 
For the word labels, we used the MS-COCO (Lin et al., 2014) 
and CIFAR-100 datasets (Krizhevsky et al., 2009). 
We evaluated the pre-trained CLIP models (Radford et al., 2021) 
by using our benchmark test. Please see more details in our paper. 

![Schematic overview of our benchmark test](DOC/figure_1.png "overview")

## Requirements

The required python libraries to run our codes in your local machine are
summarized in Pipfile, managed by pipenv. 
If you don't have pipenv, you can simply get it with ``pip install pipenv``.
Then, run ``pipenv install`` to install the requirements. 

## Run in a local environment
### download datasets and run tests

To test our benchmark test, you need to just run ``python main.py``.
Then, the code will download the image and label datasets, 
make word-superimposed images, and test a pre-trained CLIP model. 
The evaluation data will be stored in the DATA directory.

### Jupyter notebook Demo 

Once the evaluation data is created, 
you can check the output and make the figures using demo_evaluation.ipynb 
and demo_rsa.ipynb


## Run in Google Colab

ICLR2022.ipynb includes all the codes to run our benchmark test. 
You can also test it on Google Colab without environment settings. 

## Citation information of our paper

```
@inproceedings{lemesle2021evaluating,
  title={Language-biased image classification: evaluation based on semantic representations},
  author={Lemesle, Yoann and Sawayama, Masataka and Valle-Perez, Guillermo and Adolphe, Maxime and Sauz{\'e}on, H{\'e}l{\`e}ne and Oudeyer, Pierre-Yves},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```

## References of other researches the source code depends on

```
@article{radford2021learning,
  author    = {Alec Radford and
               Jong Wook Kim and
               Chris Hallacy and
               Aditya Ramesh and
               Gabriel Goh and
               Sandhini Agarwal and
               Girish Sastry and
               Amanda Askell and
               Pamela Mishkin and
               Jack Clark and
               Gretchen Krueger and
               Ilya Sutskever},
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  journal   = {CoRR},
  volume    = {abs/2103.00020},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.00020},
  archivePrefix = {arXiv},
  eprint    = {2103.00020},
  timestamp = {Thu, 04 Mar 2021 17:00:40 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-00020.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


@article{cichy2016similarity,
  title={Similarity-based fusion of MEG and fMRI reveals spatio-temporal dynamics in human cortex during visual object recognition},
  author={Cichy, Radoslaw Martin and Pantazis, Dimitrios and Oliva, Aude},
  journal={Cerebral Cortex},
  volume={26},
  number={8},
  pages={3563--3579},
  year={2016},
  publisher={Oxford University Press}
}

@article{mohsenzadeh2019reliability,
  title={Reliability and Generalizability of Similarity-Based Fusion of MEG and fMRI Data in Human Ventral and Dorsal Visual Streams},
  author={Mohsenzadeh, Yalda and Mullin, Caitlin and Lahner, Benjamin and Cichy, Radoslaw Martin and Oliva, Aude},
  journal={Vision},
  volume={3},
  number={1},
  pages={8},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}

@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}

@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey and others},
  year={2009},
  publisher={Citeseer}
}

@article{mikolov2013efficient,
  title={Efficient estimation of word representations in vector space},
  author={Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  journal={arXiv preprint arXiv:1301.3781},
  year={2013}
}

@inproceedings{mikolov2013distributed,
  title={Distributed representations of words and phrases and their compositionality},
  author={Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg S and Dean, Jeff},
  booktitle={Advances in neural information processing systems},
  pages={3111--3119},
  year={2013}
}


```