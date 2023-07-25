***
 The folder *annotations* contains multiple JSON files named with scene names, and each JSON file contains the annotation of a scene with fixed view. For each JSON,


    JSON{
        “image_shape”: {“height”:int, “width”: int},
        “frame_info”: frame_dict,
    }

    frame_dict{
        ${image_name}: [person_dict, ...],
        ...
    }

    person_dict{
        “bbox”: [x1, y1, x2, y2],
        “keypoints”: [x1, y1, …, x17, y17] or null,
        “type”: “normal’ or “severe_occlusion" or “truncation” or “ghost” (Distortion or dislocation of the body),
        "hvip2d": [x1, y1] or null,
        "hvip3d":  [float, float, float] or null,
        "torso_center": [float, float, float] or null
    }

1) ${image_name} is like "playground0_00_000000.jpg".
2) "bbox": [left, top, right, bottom]
3) xi and yi are floats ranged from 0 to 1, representing the ratio of the coordinates to the width and height of the image, respectively.
4) The value of "keypoints" is at the COCO17 format.
5) "hvip3d" is the projection point of a person’s 3D torso center on the ground plane in the global camera space.

***
The folder *params* contains the camera and ground plane parameters.

load.py is an example which can extract
* 	ground plane function Ax+By+Cz+D=0 and corresponding mask in the image space. 
	(On scene/image may contains multiple grounds and ground masks.)
* 	3×3 camera instincts