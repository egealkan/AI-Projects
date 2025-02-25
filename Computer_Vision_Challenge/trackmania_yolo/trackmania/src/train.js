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

async function loadDataset() {
  const datasetPath = path.join(process.cwd(), "dataset", "train");
  console.log("Loading dataset from:", datasetPath);

  const images = [];
  const labels = [];

  // Check if dataset directory exists
  if (!(await fs.pathExists(datasetPath))) {
    throw new Error(`Dataset directory not found: ${datasetPath}`);
  }

  const imageFiles = await fs.readdir(path.join(datasetPath, "images"));
  console.log(`Found ${imageFiles.length} images`);

  for (const imageFile of imageFiles) {
    if (!imageFile.match(/\.(jpg|jpeg|png)$/i)) continue;

    const imagePath = path.join(datasetPath, "images", imageFile);
    const labelPath = path.join(
      datasetPath,
      "labels",
      imageFile.replace(/\.[^/.]+$/, ".txt")
    );

    try {
      // Verify label file exists
      if (!(await fs.pathExists(labelPath))) {
        console.warn(`No label file found for ${imageFile}, skipping`);
        continue;
      }

      // Load and process image
      const image = await loadImage(imagePath);
      const canvas = createCanvas(640, 640);
      const ctx = canvas.getContext("2d");

      // Maintain aspect ratio while resizing
      const scale = Math.min(640 / image.width, 640 / image.height);
      const width = image.width * scale;
      const height = image.height * scale;
      const x = (640 - width) / 2;
      const y = (640 - height) / 2;

      ctx.drawImage(image, x, y, width, height);

      // Convert to tensor
      const imageData = ctx.getImageData(0, 0, 640, 640);
      const tensor = tf.tidy(() => {
        return tf.browser
          .fromPixels(imageData)
          .toFloat()
          .div(255.0)
          .expandDims();
      });

      // Load and parse label
      const labelContent = await fs.readFile(labelPath, "utf-8");
      const boxes = labelContent
        .trim()
        .split("\n")
        .map((line) => {
          const [classId, x, y, w, h] = line.split(" ").map(Number);
          return [x, y, w, h, classId];
        });

      images.push(tensor);
      labels.push(boxes);

      console.log(`Processed ${imageFile} with ${boxes.length} labels`);
    } catch (err) {
      console.error(`Error processing ${imageFile}:`, err);
    }
  }

  if (images.length === 0) {
    throw new Error("No valid training data found");
  }

  return [images, labels];
}

function createModel() {
  const model = tf.sequential();

  // Input layer
  model.add(
    tf.layers.conv2d({
      inputShape: [640, 640, 3],
      filters: 16,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
    })
  );

  // Feature extraction layers
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Detection head
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 256, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(
    tf.layers.dense({
      units: 5 * CLASSES.length, // x, y, w, h, confidence for each class
      activation: "sigmoid",
    })
  );

  return model;
}

async function train() {
  try {
    console.log("Loading dataset...");
    const [images, labels] = await loadDataset();
    console.log(`Dataset loaded: ${images.length} samples`);

    console.log("Creating model...");
    const model = createModel();

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: "meanSquaredError",
      metrics: ["accuracy"],
    });

    model.summary();

    console.log("Starting training...");
    await model.fit(tf.concat(images), tf.concat(labels), {
      epochs: 50,
      batchSize: 4,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(
            `Epoch ${epoch + 1}: ` +
              `loss = ${logs.loss.toFixed(4)}, ` +
              `accuracy = ${logs.acc.toFixed(4)}`
          );
        },
      },
    });

    console.log("Saving model...");
    await model.save("file://./models/trackmania");

    console.log("Training complete!");
  } catch (error) {
    console.error("Training failed:", error);
  }
}

// Start training if this file is run directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  train();
}

export { train };
