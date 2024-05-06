# PixaShield: AI Intelligent Camera Solution

## Introduction

PixaShield, developed by the talented team from Thakur College of Engineering, emerged as the first prize winner in the Rajasthan Police Hackathon 1.0. Our solution addresses the problem statement of creating an AI-powered intelligent camera system. PixaShield integrates cutting-edge technology, extensive datasets, and innovative features to provide a robust surveillance solution.

## Team Members

- Ishan Naik
- Ishaan Gupta
- Kunal Pawar
- Kartik Prajapati

## Problem Statement

The challenge presented was to develop an AI solution for an intelligent camera system. PixaShield not only meets this requirement but also exceeds expectations by incorporating advanced functionalities and features.

## Features

### Object Detection and Classification

PixaShield utilizes Ultralytics YOLOv8, trained on two diverse datasets:

1. **Infrared Dataset:**
   - Classes: "person", "dog", "drone", "fire", "car"
   - Number of Images: 10,000

2. **Online Sourced Dataset:**
   - Additional Classes: "Grenade", "Handgun", "Rifle", "Steel arms", "Climbing", "Fall", "Violence", "Fire"
   - Number of Images: Over 125,000

### Email Notifications

Upon detecting an object of interest, PixaShield sends an email notification using its dedicated email ID.

### Call Alerts via Twilio API

PixaShield is equipped with Twilio API integration, enabling it to initiate a call alert when detecting specific objects.

### Real-time Video Processing

The system supports real-time video processing, enabling continuous monitoring and instant response to potential threats.

https://github.com/Ishannaik/RJPOLICE_HACK_238_PIXASHIELD_3/assets/11766476/af23ac60-9d81-4111-9a37-69a761d21191

![chrome_Kl39heYFIn](https://github.com/Ishannaik/RJPOLICE_HACK_238_PIXASHIELD_3/assets/11766476/b82e7a94-48c1-461a-ad95-d1f5bdeaaa7a)

![chrome_l7aP9PbNTi](https://github.com/Ishannaik/RJPOLICE_HACK_238_PIXASHIELD_3/assets/11766476/d2bf9733-385b-4930-9e8e-882bec7f91ef)

  Ultralytics YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification, and pose estimation tasks.

<p align="center">
  <img src="https://github.com/Ishannaik/RJPOLICE_HACK_238_PIXASHIELD_3/assets/11766476/79854bec-b420-4de2-9966-2b3519befe36" alt="Image">
</p>

| Model   | Size (pixels) | mAPval | Speed CPU ONNX (ms) | Speed A100 TensorRT (ms) | Params (M) | FLOPs (B) |
|---------|---------------|--------|---------------------|-------------------------|------------|-----------|
| YOLOv8n | 640           | 37.3   | 80.4                | 0.99                    | 3.2        | 8.7       |
| YOLOv8s | 640           | 44.9   | 128.4               | 1.20                    | 11.2       | 28.6      |
| YOLOv8m | 640           | 50.2   | 234.7               | 1.83                    | 25.9       | 78.9      |
| YOLOv8l | 640           | 52.9   | 375.2               | 2.39                    | 43.7       | 165.2     |
| YOLOv8x | 640           | 53.9   | 479.1               | 3.53                    | 68.2       | 257.8     |

<p align="center">
  <img src="https://github.com/Ishannaik/RJPOLICE_HACK_238_PIXASHIELD_3/assets/11766476/dc14c402-9969-4262-9b41-7a78d993e77b" alt="Image">
</p>
