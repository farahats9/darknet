rem # How to calculate mAP (mean average precision)


rem darknet.exe detector map data/voc.data cfg/yolov2-tiny-voc.cfg yolov2-tiny-voc.weights


rem darknet.exe detector map data/voc.data cfg/yolov2-voc.cfg yolo-voc.weights


darknet.exe detector map data/obj.data cfg/yolov2-obj.cfg yolov2-obj_200.weights

pause
