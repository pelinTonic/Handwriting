for roi in rois:

    col_splits_resized = cv2.resize(roi,(32,32))
    col_splits_resized = col_splits_resized / 255.0
    col_splits_resized = np.expand_dims(col_splits_resized, axis=0)
    print(predict(col_splits_resized))