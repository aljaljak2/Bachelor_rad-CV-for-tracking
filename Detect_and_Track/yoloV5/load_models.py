import torch

def yoloV5l(imgsz=640):
    '''
    this function loads YoloV5l pytorch models from torch hub with configurable input size

    Parameters
    ----------
    imgsz : int or tuple, optional
        Input image size for YOLO model (default: 640)

    Return
    ----------
    modeli : pytorch model
        pytorch YoloV5l model
    ball_model : pytorch model
        pytorch YoloV5l model to detect the ball specifically
    '''

    # Load models without imgsz parameter
    modeli = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    modeli.classes = [0, 32]
    # Set image size after loading
    modeli.imgsz = imgsz

    ball_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    ball_model.classes = [32]
    ball_model.conf = 0.15
    ball_model.max_det = 1
    # Set image size after loading
    ball_model.imgsz = imgsz
    
    print('\n-------------------------------------------------------------------------------------------\n')

    return modeli, ball_model