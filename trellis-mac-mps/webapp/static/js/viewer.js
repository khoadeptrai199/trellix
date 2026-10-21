// Minimal three.js glTF viewer for the generated .glb assets. Exposes `window.trellisViewer`
// with a single `load(url)` method -- app.js calls this once a job finishes.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const canvas = document.getElementById("viewer-canvas");
const container = document.getElementById("viewer-container");
const emptyState = document.getElementById("viewer-empty");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070a);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(2, 1.5, 2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
keyLight.position.set(3, 5, 4);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x88aaff, 0.6);
rimLight.position.set(-4, 2, -3);
scene.add(rimLight);

let currentModel = null;
const loader = new GLTFLoader();

function resize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

function frameObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;

  object.position.sub(center);

  const dist = maxDim * 1.8;
  camera.position.set(dist, dist * 0.7, dist);
  camera.near = maxDim / 100;
  camera.far = maxDim * 100;
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();
}

function load(url) {
  emptyState.hidden = true;
  if (currentModel) {
    scene.remove(currentModel);
    currentModel = null;
  }
  resize();
  loader.load(
    url,
    (gltf) => {
      currentModel = gltf.scene;
      scene.add(currentModel);
      frameObject(currentModel);
    },
    undefined,
    (err) => {
      console.error("Failed to load glb:", err);
      emptyState.hidden = false;
      emptyState.textContent = "Failed to load 3D asset -- see console.";
    }
  );
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
resize();
animate();

window.trellisViewer = { load };
