import os
import glob
import pandas as pd
from deepface import DeepFace

# Replace your DeepFace call with an explicit detector:
result = DeepFace.analyze(
    img_path="p1XLx.png",
    actions=['age', 'gender', 'emotion', 'race'],
    detector_backend='retinaface',  # Or 'mtcnn' / 'yolov8'
    enforce_detection=False         # Prevents crashing if no face is detected
)

print(result)

def main():
    image_dir = r"C:\Users\Admin\Downloads\Test API"
    output_csv = "results.csv"
    
    print(f"Looking for images in {image_dir}...")
    # Collect all jpg and png files
    image_paths = []
    for ext in ('*.jpg'):
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    print(f"Found {len(image_paths)} images. Starting analysis...")
    
    results = []
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        try:
            # We set enforce_detection=False so the script doesn't crash if no face is found
            objs = DeepFace.analyze(
                img_path, 
                actions=['age', 'gender', 'race', 'emotion'], 
                enforce_detection=False
            )
            
            # DeepFace.analyze returns a list of dictionaries in newer versions, 
            # but a single dictionary in older versions. Let's normalize it to a list.
            if not isinstance(objs, list):
                objs = [objs]
                
            for i, obj in enumerate(objs):
                # We fetch the dominant values from the analysis dictionary
                row = {
                    "filename": filename,
                    "face_index": i,
                    "age": obj.get("age"),
                    "dominant_gender": obj.get("dominant_gender") or max(obj.get("gender", {"unknown": 1}), key=obj.get("gender", {"unknown": 1}).get) if "gender" in obj else None,
                    "dominant_race": obj.get("dominant_race"),
                    "dominant_emotion": obj.get("dominant_emotion"),
                    "error": ""
                }
                
                # In some DeepFace versions, 'dominant_gender' doesn't exist, but 'gender' does as a dictionary.
                if row["dominant_gender"] is None and "gender" in obj:
                    if isinstance(obj["gender"], dict):
                        row["dominant_gender"] = max(obj["gender"], key=obj["gender"].get)
                    elif isinstance(obj["gender"], str):
                        row["dominant_gender"] = obj["gender"]

                results.append(row)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({
                "filename": filename,
                "face_index": -1,
                "age": None,
                "dominant_gender": None,
                "dominant_race": None,
                "dominant_emotion": None,
                "error": str(e)
            })
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Analysis complete. Results saved to {output_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
