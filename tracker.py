"""
1 train Faster rcnn on full image with no resizing
## Con
-- Model was slows on the computing platform
-- Model did not prediction for some images when on the test computing platform
++ Model has potentials with quite okay accuracy that can be improved


Next steps
2. develop a much efficient model in terms of speed
 Explored
 1 Resized input image to 224X""$ Faster RCNN with resnet50 fpn
 +++ Speed improved bassed on number of images proccessed but was not enough for successful submission
 
3. Develop model meant for mobile systems 8lightweight in terms of spped of prediction)
    Explored
    1 Faster rcnn with mobilenet backbone fasterrcnn_mobilenet_v3_large_fpn inpu image resoze to 224X224
    +++ Train seems much faster
    
    
4. Tried a single step approach using SSD
    ++ model was fast and able to complete predictions  within compute time
    -- Predicted bbox were capped to 224 max hence predictions did not
    reflect the groundtruth of more than 1200 width and height
    
"""


