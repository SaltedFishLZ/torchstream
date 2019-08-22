# Metadata of Datasets

Here, we organize some common datasets' metadata into the same format, which refered to as "metaset" and can be used as a Python module directly.  
The basic componets of a meta are as follows:

## `__dataset__`

Name of the dataset

## `__layout__`

The file system layout (how directories and files are organized) of the raw dataset (following the official download guides).

## `__LABELS__`

A dict with keys being class labels (`str`) and value being the class-id (cid, `int`).

## `__SAMPLES_PER_LABEL__`

A dict with keys being class labels (`str`) and value being the sample number (interval) of each class ([low, high], `list`, inclusive).

## Dataset Filter (`TrainsetFilter`, `ValsetFilter`, `TestsetFilter`, etc)

Filters to filter samples belonging to a specified set.

## Paths

Default paths to download paths of the dataset