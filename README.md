# adversarial-nets
Code for a [blog post](http://mattdickenson.com/2017/06/18/adversarial-perturbations-in-production/) exploring adversarial perturbations

Based on [*Universal adversarial perturbations*](https://arxiv.org/abs/1610.08401) ([code](https://github.com/LTS4/universal)).

### Setup


First, set up your environment with your Google Cloud credentials according to the instructions [here](https://developers.google.com/identity/protocols/application-default-credentials).

Then, install the required packages using Anaconda:

```
$ conda env create -f environment.yml
$ source activate adversarial
# No conda-installable version of this package currently available
$ pip install --upgrade google-cloud
```

### Scripts

- *`preprocess_images.py`*: Crop and scale images from Arxiv tarball to the required size
- *`perturb.py`*: Create perturbed versions of each image
- *`label.py`*: Label images via Google Cloud Vision API
- *`analysis.py`*: Evaluate labels

### Usage

```
$ python py/perturb.py
$ python py/label.py
$ python py/analysis.py
```