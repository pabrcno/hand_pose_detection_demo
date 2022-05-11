/* eslint-disable linebreak-style */
/* eslint-disable one-var */
/* eslint-disable quotes */
/* eslint-disable max-len */
/* eslint-disable guard-for-in */
/* eslint-disable object-curly-spacing */

import "@tensorflow/tfjs-backend-webgl";
import * as mpHands from "@mediapipe/hands";
import * as params from "./params";
import * as tfjsWasm from "@tensorflow/tfjs-backend-wasm";

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`
);

import * as handdetection from "@tensorflow-models/hand-pose-detection";

import { Camera } from "./camera";
import { STATE } from "./params";

import { setBackendAndEnvFlags } from "./util";

let detector;
let camera;
let rafId;

async function createDetector() {
  switch (STATE.model) {
    case handdetection.SupportedModels.MediaPipeHands:
      const runtime = STATE.backend.split("-")[0];
      if (runtime === "mediapipe") {
        return handdetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          maxHands: STATE.modelConfig.maxNumHands,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${mpHands.VERSION}`,
        });
      } else if (runtime === "tfjs") {
        return handdetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          maxHands: STATE.modelConfig.maxNumHands,
        });
      }
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

// IMPORTANT!! key function to get handpoints
async function handsResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let hands = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateHands.

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      hands = await detector.estimateHands(camera.video, {
        flipHorizontal: false,
      });
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }
    console.log(hands);
    return hands;
  }
}

async function handsPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await handsResult();
  }

  rafId = requestAnimationFrame(handsPrediction);
}

async function app() {
  // Gui content will change depending on which model is in the query string.

  params.STATE.model = handdetection.SupportedModels.MediaPipeHands;
  const backends = params.MODEL_BACKEND_MAP[params.STATE.model];
  params.STATE.backend = backends[0];

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  handsPrediction();
}

app();
