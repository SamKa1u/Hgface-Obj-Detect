from transformers import YolosImageProcessor, YolosForObjectDetection
import cv2
"""
    detects objects in frame according to the yolo tiny size model and COCO classes

    Args:
        image (np.ndarray): The frame to be processed.
        
    Returns:
        image (np.ndarray): the processed image.
    """
def object_detect(image):
    #initialize model and image processor
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    #specify model parameters
    inputs = image_processor(
        images=image,
        return_tensors="pt"
    )
    outputs = model(**inputs)

    ## model predicts bounding boxes and corresponding COCO classes
    results = image_processor.post_process_object_detection(outputs, threshold=0.9)[0]

    #for each object detected with at least a 90% confidence score print the score, associated label,and box dimensions
    print(results)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    #annotate frame
    image = bbox(results, image, model)
    return image

"""
    annotates detected objects with bounding boxes and object labels

    Args:
        results (dict): the detected objects
        image (np.ndarray): The frame to be processed.
        model: model specifications

    Returns:
        image (np.ndarray): the labeled image.
    """
def bbox(results, image, model):
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width, channels = image.shape
    for label, box in zip(results["labels"], results["boxes"]):
        box = [round(i, 4) for i in box.tolist()]
        x1, y1, x2, y2 = box
        #scale by picture resolution
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        # print(x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(image, (x1, y2-40), (x2, y2), (255, 0, 0), -1)
        fnt_scale = text_format(model.config.id2label[label.item()], x1, x2, y2)
        cv2.putText(image, model.config.id2label[label.item()], (x1, y2-10), font, fnt_scale, (255, 255, 255), 2)
    return image

"""
    resizes text until it fits in label rectangle

    Args:
        label (): the class of object
        x1 (int): left bound of box
        y1 (int): top bound of box
        x2 (int): right bound of box
        y2 (int): bottom bound of box

    Returns:
        fnt_scale (float): the text's font scale .
    """
def text_format(label, x1, x2, y2):
    fnt_scale = 5
    t_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fnt_scale, 2)[0][0]
    t_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fnt_scale, 2)[0][1]
    if t_width < x2-x1 and t_height < y2-(y2-40):
        return fnt_scale
    else:
        while t_width > x2-x1 or t_height > (y2-(y2-40)):
            fnt_scale *= .85
            t_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fnt_scale, 2)[0][0]
            t_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fnt_scale, 2)[0][1]
        return  fnt_scale

def main():
    #camera setup
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open camera")
        exit(1)
    process = True
    while True:
        # frame error checking
        ret, frame = cam.read()
        if not ret:
            print("Could not read camera frame")
            continue
        # image = frame
        # process and annotate frame
        image = object_detect(frame)
        cv2.imshow("image", image)
        #check quit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.release()


if __name__ == "__main__":
    main()
