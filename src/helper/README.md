# Dataset Format

Each split of the dataset is structured as follows:
```
config/
    train.json
    val.json
    test.json
images/
    ACRE_train_00*.png
    ACRE_val_00*.png
    ACRE_test_00*.png
scenes/
    ACRE_train_00*.json
    ACRE_val_00*.json
    ACRE_test_00*.json
```

Each image file in the ```images``` folder has a corresponding scene description file in ```scenes``` with the same name (except for the extension).

Each ACRE problem is named after 
```
ACRE_{train/val/test}_{6_digit_problem_idx}_{2_digit_panel_idx}
```

## Configuration File

Each configuration file contains the concise scene representation for each panel. Important fields are:

* light_state: on (activated), off (inactivated), or no (no blicket machine)
* objects: a list of objects in the scene, denoted as a unique index in the shape-material-color combination space
* label: the ground-truth answer to the query
* type: type of the query
* ref: for interventional queries, it points to the original scene

## Scene Description

Scene description files contain annotations for each image. Important fields are:

* light_state: on (activated), off (inactivated), or no (no blicket machine)
* camera: camera location and orientation
* key_light: key light location
* back_light: back light location
* fill_light: fill light location
* objects: a list of objects, each of which is annotated with shape, color, size, material, bbox, and mask
* blicket_bbox (if a blicket machine): bounding box of the blicket machine
* blicket_mask (if a blicket machine): encoded mask of the blicket machine