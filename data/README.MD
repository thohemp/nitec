# Dataset preparation

Download the following datasets and place them under **nitec/data** folder.
| Dataset name | Download link                                                                                            |
|--------------|----------------------------------------------------------------------------------------------------------|
| CelebA       | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html |
| HELEN        | http://www.ifp.illinois.edu/~vuongle2/helen/                                                             |
| WIDERFACE    | http://shuoyang1213.me/WIDERFACE/                                                                        |
| Gaze360      | http://gaze360.csail.mit.edu/                                                                            |

## Folder structe
The data structure in **nitec/data** should be the following:

```md
nitec/data/

├── Gaze360
│   ├── Image
│   │   ├── test
|   │   │   ├── Face
|   │   │   │   └── ...jpg
│   │   ├── train
|   │   │   ├── Face
|   │   │   │   └── ...jpg
│   │   ├── val
|   │   │   ├── Face
|   │   │   │   └──  ...jpg
│   └── index.ts
├── HELEN
│   ├── 134212_1.jpg
│   ├── 134212_1.jpg
│   └── ...
├── img_celeba
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── Widerface
│   ├── Widerface_test/images
│   │   ├── 0 --Parade
|   │   │   ├── 0_Parade_marchingband_1_9.jpg
|   │   │   └── ...jpg
│   ├── Widerface_train/images
│   │   ├── 0 --Parade
|   │   │   └── ...jpg
│   ├── Widerface_val/images
|   │   │   0 --Parade
└── └── └── └── ...jpg
```


## NITEC Annotations

