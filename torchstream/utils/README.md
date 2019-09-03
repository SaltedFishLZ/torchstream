# Common Utilities

## metadata

This sub-module is used to collecting metadata from the datasets.

### collect

Collecting datapoints from a given dataset.

#### collect_flat

Datasets from 20BN (a company) like Sth-sth and Jester usually have the
following structure:  

```bash
Dataset Root
├── Video 0
├── Video 1
├── ...
└── Video N
```

And there are additional annotation data for each video.
It applies to the following datasets

* Something-something V1 & V2
* Jester

#### collect_folder

  Your video dataset must have the following file orgnization:

```bash
Dataset Root
├── Class 0
│   ├── Video 0
|   ├── Video 1
|   ├── ...
|   └── Video N_0
...
|
└── Class K ...
```

If you use split image frames rather than an entire video, 
{Video i} shall be a folder contain all frames in order.
for example:

```bash
├── Class 0
│   ├── Video 0
|   |   ├── 0.jpg
|   |   ├── 1.jpg
...
```

Or you can storage video files like video_0.mp4
These should be specified via [use_imgs]

This style applies to the following datasets:

* UCF101
* HMDB51
* Weizmann