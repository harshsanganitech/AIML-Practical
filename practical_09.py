import cv2
from pathlib import Path
ROOT=Path(__file__).resolve().parent
video_path=ROOT/"datasets/09_sample_cctv.mp4"
cap=cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("Could not open sample video. Put a video at datasets/09_sample_cctv.mp4 and rerun.")
    raise SystemExit(1)
subtractor=cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=32,detectShadows=False)
out_path=ROOT/"outputs/practical_09_detected.mp4"
fps=cap.get(cv2.CAP_PROP_FPS) or 10
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer=cv2.VideoWriter(str(out_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
count=0
while True:
    ok,frame=cap.read()
    if not ok: break
    mask=subtractor.apply(frame)
    _,mask=cv2.threshold(mask,220,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c)<500: continue
        x,y,bw,bh=cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),2)
        cv2.putText(frame,"Moving object",(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    writer.write(frame); count+=1
cap.release(); writer.release()
print(f"Processed {count} frames. Output saved to {out_path}")
print("Note: this demonstrates moving-object detection, not semantic identification of cars/people.")
