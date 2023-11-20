from .results import NITECResultContainer
import numpy as np
import cv2


def visualize(frame: np.ndarray, results: NITECResultContainer, confidence=0.5):
    for i in range(results.results.shape[0]):

        bbox = results.bboxes[i]
        res = results.results[i]
        
        # Extract safe min and max of x,y
        x_min=int(bbox[0])
        if x_min < 0:
            x_min = 0
        y_min=int(bbox[1])
        if y_min < 0:
            y_min = 0
        x_max=int(bbox[2])
        y_max=int(bbox[3])

        circle_center = (int(x_min +(x_max-x_min)/2.0), int(y_min +(y_max-y_min)/2))
        circle_size= int((x_max-x_min)/2.0)
        vis_frame = frame.copy()

        if res >= confidence:                    
        # cv2.rectangle(frame, (x_min, y_min) , (x_max, y_max), (0,255,0), 5)
            cv2.circle(vis_frame, circle_center, circle_size, (0,255,0),-1)
        else:
            cv2.circle(vis_frame, circle_center, circle_size,(0,0,255),-1)

            #cv2.rectangle(frame, (x_min, y_min) , (x_max, y_max), (0,0,255), 5)
        frame = cv2.addWeighted(frame,0.7,vis_frame,0.3,0)

    return frame

        
