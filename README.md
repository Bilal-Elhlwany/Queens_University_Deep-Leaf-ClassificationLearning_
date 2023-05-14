# Queens_University_Deep-Learning-ClassificationLearning_

**Dataset Description**
https://www.kaggle.com/competitions/leaf-classification/overview

The dataset consists approximately 1,584 images of leaf specimens (16 samples each of 99 species) which have been converted to binary black leaves against white backgrounds. Three sets of features are also provided per image: a shape contiguous descriptor, an interior texture histogram, and a ﬁne-scale margin histogram. For each feature, a 64-attribute vector is given per leaf sample.

Note that of the original 100 species, we have eliminated one on account of incomplete associated data in the original dataset.

**File descriptions**

1- train.csv - the training set

2- test.csv - the test set

3- sample_submission.csv - a sample submission file in the correct format

4- images - the image files (each image is named with its corresponding id)

**Data fields**

**id** - an anonymous id unique to an image

**margin_1, margin_2, margin_3, ..., margin_64** - each of the 64 attribute vectors for the margin feature

**shape_1, shape_2, shape_3, ..., shape_64** - each of the 64 attribute vectors for the shape feature

**texture_1, texture_2, texture_3, ..., texture_64** - each of the 64 attribute vectors for the texture feature
