# Advanced Automated License Plate Recognition (ALPR) System

## Overview
This project develops an advanced Automated License Plate Recognition (ALPR) system using deep learning techniques to efficiently and accurately detect and interpret vehicle license plates in real-time. The system is designed to work under various environmental conditions and can be integrated into modern traffic management solutions.

## Performance Metrics
- Achieves mAP50-95 score of 75 for license plate detection.
- Character recognition accuracy of 95.37% under controlled conditions.

<p align="center">
<a href="https://www.youtube.com/watch?v=V6HDNpW6_80">
    <img width="600" src="https://img.youtube.com/vi/V6HDNpW6_80/0.jpg" alt="Watch the video">
    </br>Watch on YouTube: Automatic number plate recognition with Yolov10, SORT and EasyOCR!
</a>
</p>

## Features
- Real-time detection and recognition of license plates from video feeds.
- High accuracy under different lighting and weather conditions.
- API for easy integration into existing traffic management systems will be considered.

## Technologies
- Deep Learning: PyTorch, TensorFlow,
- **Object Detection**: Utilized [YOLOv10](https://github.com/THU-MIG/yolov10) and [YOLOv8](https://github.com/ultralytics/ultralytics) for vehicle and license plate detection.
- **OCR Technology**: [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition from license plates.
- **Image/Video** Processing: OpenCV

## Contributors
- [David Huang](www.linkedin.com/in/davidhuang-) : Project Leader and ML Engineer
- [Edward Liu](www.linkedin.com/in/edward-liu-055438262) : Data Engineer and Model Architect
- [Anson Sun](www.linkedin.com/in/ansontsun) : ML Engineer
- [Addison Meng](https://github.com/Addisonmeng)

## Installation
Instructions for setting up the ALPR system are provided below:
```bash
git clone https://github.com/Dawae111/Advanced-ALPR-System.git
cd Advanced-ALPR-System
pip install -r requirements.txt
instructions on setting up and running the project locally.
```
## Usage
```bash
streamlit run Scripts/model_pipeline/ui.py
```
examples of how to use the system in various scenarios, along with code snippets and output images.

## Acknowledgements

This project uses [YOLOv10](https://github.com/THU-MIG/yolov10) and [YOLOv8](https://github.com/ultralytics/ultralytics) for detection tasks. We extend our gratitude to the developers of these models.

OCR capabilities are powered by [EasyOCR](https://github.com/JaidedAI/EasyOCR).

Thanks to the developers of the SORT algorithm for tracking support. [SORT GitHub](https://github.com/abewley/sort)

The pipeline design was inspired by this [repository](https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8/blob/main/README.md), from which we have adapted components.

We also acknowledge the use of the UFPR-ALPR dataset, cited in R. Laroca et al.'s paper on real-time ALPR systems.

R. Laroca, E. Severo, L. A. Zanlorensi, L. S. Oliveira, G. R. Gonçalves, W. R. Schwartz, and D. Menotti, “A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector” in 2018 International Joint Conference on Neural Networks (IJCNN), July 2018, pp. 1–10.

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.

## Additional Resources

[Video Source for Testing](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/)
