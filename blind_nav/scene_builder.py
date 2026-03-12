#--- Phase 2: Scene Builder ---#
# Extract the actual data: what was detected and where in the frame.

'''
Transforms raw YOLO detection results into a human-readable scene description string.
For each detected object, it extracts three pieces of information:
    What — the class label (person, car, truck, etc.)
    Where — horizontal position by dividing the frame into thirds
    How close — estimated proximity using the bounding box area relative to the frame area
'''

def build_scene_description(coco_results, frame_width, custom_results=None):
    """
    Build a combined scene description from COCO and custom model results.

    Args:
        coco_results:   Results from the pre-trained YOLOv8n model (always present)
        frame_width:    Width of the camera frame in pixels
        custom_results: Results from the custom-trained model (None if not loaded)
    """
    frame_height = coco_results[0].orig_shape[0]
    frame_area = frame_width * frame_height

    # Collect detections from the COCO model (person, car, truck...)
    detections = _extract_detections(coco_results, frame_width, frame_area)

    # Collect detections from the custom model (puddle, Fence, stairs …)
    if custom_results is not None:
        detections += _extract_detections(custom_results, frame_width, frame_area)

    if not detections:
        return "No objects detected nearby."

    return "Detected: " + ", ".join(detections)

def _extract_detections(results, frame_width, frame_area):
    """
    Extract detections from a single YOLO results object.
    Returns a list of formatted detection strings.
    """
    detections = []
    for box in results[0].boxes:
        # Get the class name
        class_id = int(box.cls[0])
        label = results[0].names[class_id]

        # Get confidence
        confidence = float(box.conf[0])

        # Get bounding box center x position
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2

        # Determine position: left / center / right
        if center_x < frame_width / 3:
            position = "to your left"
        elif center_x < 2 * frame_width / 3:
            position = "ahead of you"
        else:
            position = "to your right"

        # Estimate relative size (bigger box = closer)
        box_area = (x2 - x1) * (y2 - y1)
        size_ratio = box_area / frame_area

        if size_ratio > 0.25:
            proximity = "very close"
        elif size_ratio > 0.05:
            proximity = "nearby"
        else:
            proximity = "in the distance"

        # Only detections with confidence > 0.3 are included (filters noise).
        if confidence > 0.3:
            detections.append(f"{label} ({position}, {proximity})")

    return detections