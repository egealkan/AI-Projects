import * as tf from "@tensorflow/tfjs-node";
import fs from "fs-extra";
import path from "path";
import { createCanvas, loadImage } from "canvas";

const CLASSES = [
  "car",
  "wheel",
  "road",
  "barrier",
  "checkpoint",
  "finish_line",
  "tree",
  "building",
];

async function loadModel() {
  const modelPath = "file://./models/trackmania";
  return await tf.loadLayersModel(modelPath);
}

async function detectImage(imagePath) {
  // Load and preprocess image
  const image = await loadImage(imagePath);
  const canvas = createCanvas(640, 640);
  const ctx = canvas.getContext("2d");

  // Maintain aspect ratio
  const scale = Math.min(640 / image.width, 640 / image.height);
  const width = image.width * scale;
  const height = image.height * scale;
  const x = (640 - width) / 2;
  const y = (640 - height) / 2;

  ctx.drawImage(image, x, y, width, height);

  // Convert to tensor
  const imageData = ctx.getImageData(0, 0, 640, 640);
  const tensor = tf.tidy(() => {
    return tf.browser.fromPixels(imageData).toFloat().div(255.0).expandDims();
  });

  // Load model and run prediction
  const model = await loadModel();
  const predictions = await model.predict(tensor).array();

  // Process predictions
  const detections = [];
  for (let i = 0; i < predictions[0].length; i += 5) {
    const [x, y, w, h, confidence] = predictions[0].slice(i, i + 5);
    const classIndex = Math.floor(i / 5);

    if (confidence > 0.5) {
      detections.push({
        class: CLASSES[classIndex],
        confidence,
        bbox: [x, y, w, h],
      });
    }
  }

  return detections;
}

async function detect() {
  try {
    const screenshotsDir = path.join(process.cwd(), "screenshots");
    const files = await fs.readdir(screenshotsDir);

    for (const file of files) {
      if (!file.match(/\.(jpg|jpeg|png)$/i)) continue;

      console.log(`\nProcessing ${file}...`);
      const imagePath = path.join(screenshotsDir, file);

      const detections = await detectImage(imagePath);

      console.log("Detections:");
      detections.forEach((det) => {
        console.log(
          `- ${det.class}: ${(det.confidence * 100).toFixed(1)}% confidence`
        );
      });
    }
  } catch (error) {
    console.error("Detection failed:", error);
  }
}

// Start detection if this file is run directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  detect();
}

export { detect };
