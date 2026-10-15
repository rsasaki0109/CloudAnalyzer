// Eye-Dome Lighting: a screen-space pass that darkens pixels lying behind
// their neighbours (in log depth), which makes the shape of unlit point
// clouds readable. Same idea as Potree/CloudCompare EDL.

import * as THREE from "three";

const vertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const fragmentShader = /* glsl */ `
  #include <packing>
  uniform sampler2D tColor;
  uniform sampler2D tDepth;
  uniform vec2 texel;
  uniform float cameraNear;
  uniform float cameraFar;
  uniform float strength;
  uniform float radius;
  varying vec2 vUv;

  // log2 of a measured distance along the view axis.
  float logDepth(float d) {
    float viewZ = perspectiveDepthToViewZ(d, cameraNear, cameraFar);
    return log2(-viewZ);
  }

  // Render targets hold linear color; convert for the screen like three.js does.
  void finish(vec4 color) {
    gl_FragColor = color;
    #include <colorspace_fragment>
  }

  void main() {
    vec4 color = texture2D(tColor, vUv);
    float sampleDepth = textureLod(tDepth, vUv, 0.0).x;
    if (sampleDepth >= 1.0) {
      finish(color);
      return;
    }
    float depth = logDepth(sampleDepth);
    float response = 0.0;
    for (int i = 0; i < 8; i++) {
      float angle = float(i) * 0.78539816;
      vec2 offset = vec2(cos(angle), sin(angle)) * radius * texel;
      // Explicit LOD: sampled in a loop, where derivatives are undefined.
      float neighbour = textureLod(tDepth, vUv + offset, 0.0).x;
      // Empty pixels are not nearer surfaces. Keep isolated returns visible.
      if (neighbour < 1.0) response += max(0.0, depth - logDepth(neighbour));
    }
    response /= 8.0;
    float shade = exp(-response * 300.0 * strength);
    finish(vec4(color.rgb * shade, color.a));
  }
`;

export class EdlPass {
  readonly target: THREE.WebGLRenderTarget;
  private readonly material: THREE.ShaderMaterial;
  private readonly quad: THREE.Mesh;
  private readonly screen = new THREE.Scene();
  private readonly screenCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  constructor() {
    this.target = new THREE.WebGLRenderTarget(1, 1, {
      depthTexture: new THREE.DepthTexture(1, 1, THREE.UnsignedIntType),
      depthBuffer: true,
    });
    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        tColor: { value: this.target.texture },
        tDepth: { value: this.target.depthTexture },
        texel: { value: new THREE.Vector2(1, 1) },
        cameraNear: { value: 0.1 },
        cameraFar: { value: 1000 },
        strength: { value: 1 },
        radius: { value: 1.4 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material);
    this.quad.frustumCulled = false;
    this.screen.add(this.quad);
  }

  setSize(width: number, height: number): void {
    this.target.setSize(width, height);
    (this.material.uniforms.texel.value as THREE.Vector2).set(1 / width, 1 / height);
  }

  set strength(value: number) {
    this.material.uniforms.strength.value = value;
  }

  /** Render `scene` offscreen, then composite it with EDL shading. */
  render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera): void {
    this.material.uniforms.cameraNear.value = camera.near;
    this.material.uniforms.cameraFar.value = camera.far;
    renderer.setRenderTarget(this.target);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.render(this.screen, this.screenCamera);
  }
}
