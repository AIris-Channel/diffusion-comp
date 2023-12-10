

def get_face_image(face_model, image):
    bboxes, kpss = face_model.det_model.detect(np.array(image)[:,:,::-1], max_num=1, metric='default')
    if bboxes.shape[0] == 0:
        return None
    best_bbox = bboxes[0, 0:4]
    best_score = bboxes[0, 4]
    for i in range(1, bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        if det_score > best_score:
            best_bbox = bbox
            best_score = det_score
    return image.crop(best_bbox)
