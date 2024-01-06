import cv2

def visiualize(layout, sequence):
    # 获得layout的尺寸
    width = layout.shape[1]
    height = layout.shape[0]

    cv2.imshow('layout', layout)
    cv2.waitKey(0)

