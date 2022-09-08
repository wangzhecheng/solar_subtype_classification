# solar_subtype_classification
Solar panel subtype classification with deep learning and satellite images

The model takes a satellite image of a solar installation as input and predict whether it is one of the five classes: utility-scale PV, commercial PV, residential PV, solar water heating, and negative.

To download model checkpoint, run the following command to download the ZIP file right under the code repo directory:

```
$ curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/type/checkpoint.zip
```

Unzip it such that the directory structure looks like:

```
solar_subtype_classification/checkpoint/...
```

For model training with hyperparameter tuning, please run:
```
$ python hp_search_type.py
```

For deploying the model to satellite images, please run:
```
$ python predict_type_5classes.py
```
