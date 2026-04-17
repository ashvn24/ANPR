from roboflow import Roboflow

rf = Roboflow(api_key="BnLEizmCCgcrqJUkATlR")  # get free key at roboflow.com

# 1. Best general plate detector (10k images)
proj = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
proj.version(4).download("yolov8", location="datasets/plates_global")

# 2. Low-light plates (matches your night camera)
proj2 = rf.workspace("lowlight-images").project("low-light-license-plate")
proj2.version(1).download("yolov8", location="datasets/plates_lowlight")
