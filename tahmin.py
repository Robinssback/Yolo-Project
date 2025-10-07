from ultralytics import YOLO

model_path = r"C:/Users/melis/OneDrive/Masaüstü/yoloproject/runs/detect/train4/weights/best.pt"
model = YOLO(model_path)


img_path = r"C:/Users/melis/OneDrive/Masaüstü/yoloproject/bitkig.jpg" #You may change the photo from here
results = model(img_path)

# output folder
output_dir = "runs/detect/predict/"

# import the model
model = YOLO(model_path)

# prediction
results = model.predict(
    source=img_path,
    conf=0.25,      
    save=True,     
    save_txt=True   
)
print(f"The prediction is complete the outputs are in '{output_dir}' folder.")
