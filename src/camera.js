/* eslint-disable linebreak-style */
/* eslint-disable quotes */
/* eslint-disable max-len */
/* eslint-disable guard-for-in */
/* eslint-disable object-curly-spacing */

import * as scatter from "scatter-gl";

import * as params from "./shared/params";
import { isMobile } from "./shared/util";

function createScatterGLContext(selectors) {
  const scatterGLEl = document.querySelector(selectors);
  return {
    scatterGLEl,
    scatterGL: new scatter.ScatterGL(scatterGLEl, {
      rotateOnStart: true,
      selectEnabled: false,
      styles: { polyline: { defaultOpacity: 1, deselectedOpacity: 1 } },
    }),
    scatterGLHasInitialized: false,
  };
}

const scatterGLCtxtLeftHand = createScatterGLContext(
  "#scatter-gl-container-left"
);
const scatterGLCtxtRightHand = createScatterGLContext(
  "#scatter-gl-container-right"
);

export class Camera {
  constructor() {
    this.video = document.getElementById("video");
    this.canvas = document.getElementById("output");
    this.ctx = this.canvas.getContext("2d");
  }

  /**
   * Initiate a Camera instance and wait for the camera stream to be ready.
   * @param cameraParam From app `STATE.camera`.
   */
  static async setupCamera(cameraParam) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "Browser API navigator.mediaDevices.getUserMedia not available"
      );
    }

    const { targetFPS, sizeOption } = cameraParam;
    const $size = params.VIDEO_SIZE[sizeOption];
    const videoConfig = {
      audio: false,
      video: {
        facingMode: "user",
        // Only setting the video to a specified size for large screen, on
        // mobile devices accept the default size.
        width: isMobile() ? params.VIDEO_SIZE["360 X 270"].width : $size.width,
        height: isMobile()
          ? params.VIDEO_SIZE["360 X 270"].height
          : $size.height,
        frameRate: {
          ideal: targetFPS,
        },
      },
    };

    const stream = await navigator.mediaDevices.getUserMedia(videoConfig);

    const camera = new Camera();
    camera.video.srcObject = stream;

    await new Promise((resolve) => {
      camera.video.onloadedmetadata = () => {
        resolve(video);
      };
    });

    camera.video.play();

    const videoWidth = camera.video.videoWidth;
    const videoHeight = camera.video.videoHeight;
    // Must set below two lines, otherwise video element doesn't show.
    camera.video.width = videoWidth;
    camera.video.height = videoHeight;

    camera.canvas.width = videoWidth;
    camera.canvas.height = videoHeight;
    const canvasContainer = document.querySelector(".canvas-wrapper");
    canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

    // Because the image from camera is mirrored, need to flip horizontally.
    camera.ctx.translate(camera.video.videoWidth, 0);
    camera.ctx.scale(-1, 1);

    for (const ctxt of [scatterGLCtxtLeftHand, scatterGLCtxtRightHand]) {
      ctxt.scatterGLEl.style = `width: ${videoWidth / 2}px; height: ${
        videoHeight / 2
      }px;`;
      ctxt.scatterGL.resize();

      ctxt.scatterGLEl.style.display = params.STATE.modelConfig.render3D
        ? "inline-block"
        : "none";
    }

    return camera;
  }
}
