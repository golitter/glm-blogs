import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutlinePass } from "three/addons/postprocessing/OutlinePass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

import { blogTree, markdownCount, recentFiles, type BlogFile, type BlogTreeNode } from "@/generated/blog-data";
import { PROFILE_URL, CSDN_URL, REPO_URL } from "@/lib/constants";
import { shelfLayout } from "@/lib/shelf-layout";
import { searchFiles, nodes, updateInfo, type Route, type Quality } from "@/lib/blog-experience";

type Interactive = {
  root: THREE.Object3D;
  action: () => void;
  baseScale: THREE.Vector3;
  basePosition: THREE.Vector3;
  name: string;
  glow?: THREE.MeshToonMaterial;
};

type SceneOptions = {
  onReady: () => void;
  onSearchRequest: () => void;
  quality: Quality;
  onRouteChange: (route: Route, replace?: boolean) => void;
  onInteraction: () => void;
  onHover: (name: string) => void;
  onError: () => void;
};
type DirectoryEntry = BlogFile & { folder?: BlogTreeNode };

const PALETTE = {
  cream: 0xfffaea,
  paper: 0xfffdf5,
  sage: 0xb8d493,
  sageDark: 0x66844d,
  gold: 0xd8b969,
  blush: 0xecc8c3,
  brown: 0x6b5d4c,
  ink: "#494338",
};

function flattenNode(node: BlogTreeNode): BlogFile[] {
  return [...node.files, ...node.children.flatMap(flattenNode)];
}

export function createImmersiveBlog(host: HTMLElement, options: SceneOptions) {
  const quality = options.quality;
  const low = quality === "eco";
  const pixelRatio = Math.min(devicePixelRatio, quality === "high" ? 1.7 : low ? 1 : 1.25);
  let currentRoute: Route = {view:"home"};
  let suppressRoute = false;
  let lastActivity = performance.now();
  function announce(route: Route) {
    currentRoute = route;
    if (route.view !== "category" && route.view !== "search") renderer.domElement.setAttribute("aria-label", `${route.view === "recent" ? "最近更新" : route.view === "shelf" ? "知识目录" : "全景"}，拖动可 360° 环绕，滚轮缩放。`);
    if (!suppressRoute) options.onRouteChange(route);
  }
  const cabinet = shelfLayout<BlogTreeNode>(blogTree);
  const sceneHeight = Math.max(17, cabinet.height + 4);
  const compact = matchMedia("(max-width: 700px)");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const renderer = new THREE.WebGLRenderer({ antialias: !low, powerPreference: low ? "low-power" : "high-performance" });
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(host.clientWidth, host.clientHeight);
  renderer.shadowMap.enabled = quality === "high";
  // Only the bird's sub-texel bob animates continuously, so the shadow map is rendered once and
  // refreshed on demand (hover pulls, panel show/hide) instead of every frame.
  renderer.shadowMap.autoUpdate = false;
  renderer.shadowMap.needsUpdate = true;
  renderer.localClippingEnabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1;
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute("aria-label", "三维博客场景，可拖动旋转、滚轮缩放并点击物件");
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xc9dcd5);
  const camera = new THREE.PerspectiveCamera(38, host.clientWidth / host.clientHeight, 0.1, 250);
  camera.position.set(0, 8.2, compact.matches ? 31 : 24);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 3.2, -1.5);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 5;
  controls.maxDistance = 110;
  controls.enablePan = false;
  controls.minAzimuthAngle = -Infinity;
  controls.maxAzimuthAngle = Infinity;
  controls.minPolarAngle = .18;
  controls.maxPolarAngle = Math.PI * .65;
  controls.autoRotate = false;
  controls.autoRotateSpeed = 0.16;

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const outline = new OutlinePass(new THREE.Vector2(host.clientWidth, host.clientHeight), scene, camera);
  outline.edgeStrength = 3.2;
  outline.edgeGlow = 0.35;
  outline.edgeThickness = 1.2;
  outline.visibleEdgeColor.set(0xffe4a0);
  outline.hiddenEdgeColor.set(0x71865a);
  composer.addPass(outline);
  outline.enabled = !low;
  const bloom = new UnrealBloomPass(new THREE.Vector2(host.clientWidth, host.clientHeight), 0.06, 0.32, 1.15);
  bloom.enabled = quality === "high";
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  const gradient = new THREE.DataTexture(new Uint8Array([80, 150, 210, 255]), 4, 1, THREE.RedFormat);
  gradient.minFilter = gradient.magFilter = THREE.NearestFilter;
  gradient.needsUpdate = true;
  const materialCache = new Map<number, THREE.MeshToonMaterial>();
  function toon(color: number) {
    if (!materialCache.has(color)) materialCache.set(color, new THREE.MeshToonMaterial({ color, gradientMap: gradient }));
    return materialCache.get(color)!;
  }
  const goldMaterial = new THREE.MeshPhysicalMaterial({ color: PALETTE.gold, metalness: 0.68, roughness: 0.25, clearcoat: 0.5 });
  // Beacon and hover-glow materials pulse in unison, so share one instance per colour instead of one per object.
  const beaconFresh = new THREE.MeshBasicMaterial({ color: 0xe7bd62, transparent: true, opacity: .65 });
  const beaconCalm = new THREE.MeshBasicMaterial({ color: 0xfff7d5, transparent: true, opacity: .65 });
  const glowCache = new Map<THREE.MeshToonMaterial, THREE.MeshToonMaterial>();
  function glowFor(base: THREE.MeshToonMaterial) {
    let glow = glowCache.get(base);
    if (!glow) {
      glow = base.clone();
      glow.emissive.set(0xe8c27b);
      glow.emissiveIntensity = .035;
      glowCache.set(base, glow);
    }
    return glow;
  }

  function mesh(geometry: THREE.BufferGeometry, material: THREE.Material, parent: THREE.Object3D, position: [number, number, number]) {
    const item = new THREE.Mesh(geometry, material);
    item.position.set(...position);
    item.castShadow = true;
    item.receiveShadow = true;
    parent.add(item);
    return item;
  }
  // Identical parameter sets repeat throughout the room (boards, seams, book stripes, pots), so
  // geometries are cached by key instead of rebuilt for every mesh.
  const geometryCache = new Map<string, THREE.BufferGeometry>();
  // Cached geometries are shared and outlive individual panels; clearPanel must not dispose them.
  const sharedGeometries = new Set<THREE.BufferGeometry>();
  function cached<T extends THREE.BufferGeometry>(key: string, build: () => T) {
    let geometry = geometryCache.get(key) as T | undefined;
    if (!geometry) { geometry = build(); geometryCache.set(key, geometry); sharedGeometries.add(geometry); }
    return geometry;
  }
  function rounded(parent: THREE.Object3D, size: [number, number, number], position: [number, number, number], color: number, radius = 0.12) {
    const segments = low ? 1 : 4;
    const geometry = cached(`r:${size.join(",")};${segments};${radius}`, () => new RoundedBoxGeometry(...size, segments, radius));
    return mesh(geometry, toon(color), parent, position);
  }
  // Every sphere in a given quality tier has identical segments, so one unit sphere is reused (scaled per mesh).
  function sphere(parent: THREE.Object3D, position: [number, number, number], scale: [number, number, number], color: number) {
    const width = low ? 10 : compact.matches ? 16 : 28, height = low ? 8 : 18;
    const geometry = cached(`s:${width}x${height}`, () => new THREE.SphereGeometry(1, width, height));
    const item = mesh(geometry, toon(color), parent, position);
    item.scale.set(...scale);
    return item;
  }
  function textPlane(text: string, width: number, height: number, options: { color?: string; weight?: number; align?: CanvasTextAlign; background?: string; border?: string; emphasis?: string } = {}) {
    const canvas = document.createElement("canvas");
    // Texture density tracks the plane's on-screen size; small labels skip the full 1024-wide canvas.
    canvas.width = Math.max(256, Math.min(1024, Math.round(1024 * width / 9.8)));
    canvas.height = Math.round(canvas.width * height / width);
    const fit = canvas.width - 74;
    const context = canvas.getContext("2d")!;
    if (options.background) {
      context.fillStyle = options.background;
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
    if (options.border) {
      context.strokeStyle = options.border;
      context.lineWidth = 8;
      context.strokeRect(4, 4, canvas.width - 8, canvas.height - 8);
    }
    const rawLines = text.split("\n");
    let size = canvas.height * (rawLines.length > 1 ? .32 : .64);
    const font = () => `${options.weight ?? 600} ${size}px "Microsoft YaHei", "PingFang SC", sans-serif`;
    context.font = font();
    while (rawLines.some(line => context.measureText(line).width > fit) && size > canvas.height * .24) {
      size *= .94;
      context.font = font();
    }
    context.textAlign = options.align ?? "center";
    context.textBaseline = "middle";
    context.fillStyle = options.color ?? PALETTE.ink;
    const lines = rawLines.map(line => {
      if (context.measureText(line).width <= fit) return line;
      while (line.length && context.measureText(line + "…").width > fit) line = line.slice(0, -1);
      return line + "…";
    });
    const lineHeight = size * 1.08;
    lines.forEach((line, index) => {
      const y = canvas.height / 2 + (index - (lines.length - 1) / 2) * lineHeight;
      const at = options.emphasis ? line.indexOf(options.emphasis) : -1;
      if (at >= 0) {
        const boldFont = `700 ${size}px "Microsoft YaHei", "PingFang SC", sans-serif`;
        const parts = [line.slice(0, at), options.emphasis!, line.slice(at + options.emphasis!.length)];
        const widths = parts.map((part, i) => { context.font = i === 1 ? boldFont : font(); return context.measureText(part).width; });
        let x = (canvas.width - widths.reduce((sum, value) => sum + value, 0)) / 2;
        const restore = context.textAlign;
        context.textAlign = "left";
        parts.forEach((part, i) => {
          context.font = i === 1 ? boldFont : font();
          context.fillStyle = i === 1 ? "#241f16" : options.color ?? PALETTE.ink;
          context.fillText(part, x, y);
          x += widths[i];
        });
        context.textAlign = restore;
        context.fillStyle = options.color ?? PALETTE.ink;
        return;
      }
      context.fillText(line, context.textAlign === "left" ? 38 : canvas.width / 2, y, canvas.width - 70);
    });
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, side: THREE.FrontSide, toneMapped: false, fog: false });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(width, height), material);
    plane.userData.textTexture = texture;
    plane.userData.label = text;
    return plane;
  }

  const interactives: Interactive[] = [];
  const panelInteractives = new Set<Interactive>();
  function interactive(root: THREE.Object3D, action: () => void, panel = false) {
    const labels: string[] = [];
    root.traverse(object => { if (object.userData.label) labels.push(object.userData.label); });
    let glow: THREE.MeshToonMaterial | undefined;
    if (!panel) root.traverse(object => {
      if (!glow && object instanceof THREE.Mesh && object.material instanceof THREE.MeshToonMaterial) {
        glow = glowFor(object.material); object.material = glow;
      }
    });
    const record = { root, action, baseScale: root.scale.clone(), basePosition: root.position.clone(), name: root.userData.name ?? labels[0] ?? "查看", glow };
    root.traverse((object) => { object.userData.hit = record; });
    interactives.push(record);
    if (panel) panelInteractives.add(record);
    return record;
  }

  scene.add(new THREE.HemisphereLight(0xffffff, 0xc2baa8, 1.5));
  const sun = new THREE.DirectionalLight(0xfff1d6, 1.8);
  sun.position.set(-10, 20, 14);
  sun.castShadow = true;
  sun.shadow.mapSize.set(compact.matches ? 1024 : 2048, compact.matches ? 1024 : 2048);
  Object.assign(sun.shadow.camera, { left: -19, right: 19, top: 17, bottom: -12, near: 1, far: 60 });
  sun.shadow.bias = -0.0001;
  sun.shadow.normalBias = 0.08;
  scene.add(sun);
  const fill = new THREE.PointLight(0xbde9d1, 9, 32, 1.6);
  fill.position.set(9, 9, 10);
  scene.add(fill);
  const rearLight = new THREE.DirectionalLight(0xe5efff, 1.2);
  rearLight.position.set(3, 12, -18);
  scene.add(rearLight);

  const world = new THREE.Group();
  scene.add(world);
  // Cloud island and translucent backdrop.
  const island = mesh(new THREE.CylinderGeometry(12.6, 11.8, 0.8, 64), toon(0xd4dfba), world, [0, -0.5, -1]);
  island.scale.z = 0.72;
  mesh(new THREE.CylinderGeometry(12.2, 12.2, 0.16, 64), toon(PALETTE.cream), world, [0, -0.02, -1]).scale.z = 0.72;
  for (let i = 0; i < 24; i++) {
    const angle = i / 24 * Math.PI * 2;
    sphere(world, [Math.cos(angle) * 11.6, -0.75 + Math.sin(i * 2.7) * 0.12, -1 + Math.sin(angle) * 7.1], [1.4 + i % 3 * .16, .78, 1.05], 0xfffdf5);
  }
  // Architectural arches, open to the sky, frame the miniature room.
  for (const x of [-7.3, 0, 7.3]) {
    rounded(world, [6.8, 6.8, .5], [x, 3.4, -6], 0x9eb6a0, .12);
    const arch = mesh(new THREE.TorusGeometry(3.4, .22, 10, 48, Math.PI), toon(PALETTE.cream), world, [x, 6.8, -5.7]);
    arch.castShadow = true;
    for (const side of [-1, 1]) rounded(world, [.44, 6.8, .5], [x + side * 3.4, 3.4, -5.7], PALETTE.cream);
    rounded(world, [6.7, .22, .4], [x, 6.75, -5.7], PALETTE.cream);
    rounded(world, [.16, 3.1, .22], [x, 8.3, -5.7], PALETTE.gold, .04);
  }
  // Inlaid floor seams make depth and the scale of the room visible.
  for (let z = -5; z <= 5; z += 1.25) {
    const seam = rounded(world, [18, .015, .035], [0, .09, z], 0xd7ccb5, .005);
    seam.castShadow = false;
  }
  const rug = mesh(new THREE.CylinderGeometry(4.9, 4.9, .035, 64), toon(0xb6c9a1), world, [0, .12, 2.4]);
  rug.scale.z = .63;
  const rugTrim = mesh(new THREE.TorusGeometry(4.65, .035, 8, 96), toon(PALETTE.cream), world, [0, .15, 2.4]);
  rugTrim.rotation.x = -Math.PI / 2;
  rugTrim.scale.y = .63;
  function plant(x: number, z: number, scale = 1) {
    const pot = new THREE.Group();
    pot.position.set(x, .1, z);
    pot.scale.setScalar(scale);
    world.add(pot);
    mesh(cached("pot", () => new THREE.CylinderGeometry(.53, .37, .85, 24)), toon(PALETTE.blush), pot, [0, .43, 0]);
    mesh(cached("pot-rim", () => new THREE.CylinderGeometry(.48, .48, .06, 24)), toon(PALETTE.brown), pot, [0, .86, 0]);
    for (let i = 0; i < 8; i++) {
      const a = i * 2.4;
      const leaf = sphere(pot, [Math.cos(a) * .4, 1.3 + i * .13, Math.sin(a) * .4], [.2, .68, .13], i % 2 ? 0x789768 : 0xa6c182);
      leaf.rotation.set(Math.sin(a) * .55, a, Math.cos(a) * .55);
    }
  }
  plant(-10.6, 1.6, 1.1);
  plant(10.6, 1.6, 1.1);
  plant(-4.8, -4.1, .8);
  plant(4.8, -4.1, .8);
  plant(-5.8, -7, .9);
  plant(5.8, -7, .9);
  // The rear of the room has its own timber panels, trim and maker's plaque.
  for (const x of [-7.3, 0, 7.3]) {
    for (let offset = -3; offset <= 3; offset += .75) {
      rounded(world, [.035, 6.3, .06], [x + offset, 3.4, -6.28], 0x7e9b82, .01);
    }
    for (const y of [.35, 6.5]) rounded(world, [6.8, .18, .14], [x, y, -6.32], PALETTE.cream, .035);
  }
  const rearPlaque = rounded(world, [5, 1.5, .18], [0, 4.6, -6.4], PALETTE.cream);
  const rearSignature = textPlane("Golemon Blogs", 4.5, .7, {color: "#566d43"});
  rearSignature.position.z = -.11;
  rearSignature.rotation.y = Math.PI;
  rearPlaque.add(rearSignature);
  // Warm hanging lights and a little garland, built from actual geometry.
  const bulbMaterial = new THREE.MeshBasicMaterial({color: 0xffe6ac});
  for (const x of [-10, 10]) {
    rounded(world, [.07, 3, .07], [x, 10, -3.6], PALETTE.gold, .02);
    mesh(cached("lampshade", () => new THREE.ConeGeometry(.8, .65, 32, 1, true)), toon(PALETTE.sage), world, [x, 8.3, -3.6]);
    mesh(cached("bulb", () => new THREE.SphereGeometry(.19, 16, 12)), bulbMaterial, world, [x, 8.05, -3.6]);
  }
  // The balloon and its tether share the room's coordinates, attached to the right lamp post.
  const repoBalloon = new THREE.Group();
  repoBalloon.position.set(10, 9, -3.6);
  world.add(repoBalloon);
  // Matte fabric panels echo the cream arches and sage book covers.
  const balloonCenter = new THREE.Vector3(1.55, 2.15, 0);
  for (let panelIndex = 0; panelIndex < 12; panelIndex++) {
    const panelGeometry = new THREE.SphereGeometry(1, 3, 12, panelIndex * Math.PI / 6, Math.PI / 6);
    const fabric = mesh(panelGeometry, toon(panelIndex % 3 === 0 ? 0xb6c99d : 0xeee5c9), repoBalloon, balloonCenter.toArray() as [number,number,number]);
    fabric.scale.set(1.12, 1.36, 1.02);
  }
  mesh(new THREE.ConeGeometry(.12, .19, 8), toon(PALETTE.sage), repoBalloon, [1.55,.75,0]);
  for (const sign of [-1,1]) {
    const ribbon = sphere(repoBalloon,[1.55 + sign * .17,.7,.12],[.22,.11,.07],PALETTE.sage);
    ribbon.rotation.z = sign * .35;
  }
  const tether = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0,0,0), new THREE.Vector3(.48,-.2,0),
    new THREE.Vector3(1.05,.1,0), new THREE.Vector3(1.55,.72,0),
  ]);
  mesh(new THREE.TubeGeometry(tether,24,.022,5,false), goldMaterial, repoBalloon, [0,0,0]);
  const tie = mesh(new THREE.TorusGeometry(.1,.026,8,16), goldMaterial, repoBalloon, [0,0,0]);
  tie.rotation.x = Math.PI / 2;
  // Links are actual hanging tags below the balloon, not buttons pasted on its skin.
  for (const x of [1.05,2.05]) {
    mesh(new THREE.CylinderGeometry(.018,.018,1.18,5),goldMaterial,repoBalloon,[x,.08,.12]);
  }
  for (const [label,url,y] of [["GitHub 仓库",REPO_URL,.27],["Issues 留言",`${REPO_URL}/issues`,-.3]] as const) {
    const link = new THREE.Group(); link.position.set(1.55,y,.16); link.userData.name = label; repoBalloon.add(link);
    rounded(link,[2.08,.47,.14],[0,0,0],label.startsWith("GitHub") ? PALETTE.paper : 0xd4dfba,.09);
    for (const side of [1,-1]) {
      const text = textPlane(label,1.8,.32,{color:"#53633d",weight:600}); text.position.z = side * .08; text.rotation.y = side === 1 ? 0 : Math.PI; link.add(text);
    }
    interactive(link, () => window.open(url,"_blank","noopener,noreferrer"));
  }  const garlandPoints = Array.from({length: 33}, (_, i) => new THREE.Vector3(-10 + i * .625, 10.9 - Math.sin(i / 32 * Math.PI) * 1.2, -5.1));
  mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(garlandPoints), 48, .025, 5, false), goldMaterial, world, [0, 0, 0]);
  for (let i = 0; i < 13; i++) {
    const x = -9.5 + i * 19 / 12;
    sphere(world, [x, 10.9 - Math.sin((x + 10) / 20 * Math.PI) * 1.2, -5.1], [.075, .095, .075], 0xffe3a3);
  }

  // World title and fully 3D navigation plaques.
  const titleBoard = rounded(world, [10.8, 2.1, .35], [0, Math.max(11.7, cabinet.height + 1.8), -5.4], PALETTE.paper, .18);
  const title = textPlane("Golemon Blogs", 9.8, 1.05, { weight: 700, color: "#566d43" });
  title.position.set(0, .4, .19);
  titleBoard.add(title);
  const backTitle = textPlane("Golemon Blogs", 9.8, 1.05, {weight: 700, color: "#566d43"});
  backTitle.position.set(0, .4, -.19);
  backTitle.rotation.y = Math.PI;
  titleBoard.add(backTitle);
  const description = textPlane("记录大模型、智能体与系统工程相关的学习与实践。", 9.8, .48, { weight: 600, color: "#687a51" });
  description.position.set(0, -.24, .2);
  titleBoard.add(description);
  const backDescription = description.clone();
  backDescription.position.z = -.2;
  backDescription.rotation.y = Math.PI;
  titleBoard.add(backDescription);
  const subtitle = textPlane(`${markdownCount} 篇笔记 · 拖动环绕 360° · 点击书本阅读`, 8.3, .32, { weight: 400, color: "#998d71", emphasis: `${markdownCount} 篇笔记` });
  subtitle.position.set(0, -.72, .2);
  titleBoard.add(subtitle);

  let desiredPosition = camera.position.clone();
  let desiredTarget = controls.target.clone();
  let movingCamera = false;
  let currentView: "home" | "shelf" | "recent" | "panel" = compact.matches ? "shelf" : "home";
  function goTo(position: THREE.Vector3, target: THREE.Vector3) {
    lastActivity = performance.now();
    desiredPosition = position;
    desiredTarget = target;
    movingCamera = true;
    controls.enabled = false;
    controls.autoRotate = false;
  }
  function fitView(target: THREE.Vector3, width: number, height: number, tilt = 0) {
    const distance = Math.max(height, width / camera.aspect) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.18;
    controls.maxDistance = Math.max(110, distance * 2);
    camera.far = Math.max(250, distance * 4);
    camera.updateProjectionMatrix();
    goTo(target.clone().add(new THREE.Vector3(0, tilt * distance, distance)), target);
  }
  const overviewTarget = new THREE.Vector3(0, 5 + (sceneHeight - 17) / 2, -1);
  const overview = () => fitView(overviewTarget.clone(), 27, sceneHeight, .3);
  const homeView = () => { currentView = "home"; announce({view:"home"}); overview(); };
  const shelfView = () => { currentView = "shelf"; announce({view:"shelf"}); fitView(new THREE.Vector3(-7.1, cabinet.height / 2 + .55, -2.2), 10, cabinet.height + 2, .07); };
  const recentView = () => { currentView = "recent"; announce({view:"recent"}); fitView(new THREE.Vector3(7.2, 4.3, -2.2), 9.4, 10, .04); };
  // Left: category bookshelf with physical volumes.
  const shelf = new THREE.Group();
  shelf.position.set(-7.1, .35, -2.5);
  shelf.rotation.y = .08;
  world.add(shelf);
  rounded(shelf, [8.4, cabinet.height, .45], [0, cabinet.height / 2, -.65], 0xcbb991, .15);
  rounded(shelf, [7.9, cabinet.height - .45, .32], [0, cabinet.height / 2, -.39], 0xf5eedb, .12);
  // Full-depth side panels, projecting shelves and brass feet read as furniture.
  for (const x of [-4.12, 4.12]) {
    rounded(shelf, [.28, cabinet.height, 1.35], [x, cabinet.height / 2, -.04], 0xb8a37a, .06);
    rounded(shelf, [.32, .32, 1.3], [x, -.1, -.04], PALETTE.gold, .05);
  }
  for (const y of cabinet.boards) rounded(shelf, [8.65, .22, 1.5], [0, y, .04], 0xb8a37a, .06);
  for (const y of cabinet.boards) {
    rounded(shelf, [8.05, .15, .15], [0, y, -.92], 0xb8a37a, .025);
    for (const x of [-3.85, 3.85]) sphere(shelf, [x, y, -1.02], [.055, .055, .025], PALETTE.gold);
  }
  const shelfContents = new THREE.Group();
  shelf.add(shelfContents);
  const bookColors = [0xb9d594, 0xe9c4bd, 0xe6cf89, 0xc9b9d9, 0xaecfd1, 0xc6c5b5];
  const shelfHeader = textPlane(`知识目录 · ${blogTree.length} 个分类`, 6.8, .55, {color: "#5e754a"});
  shelfHeader.position.set(0, cabinet.height - .48, .2);
  shelfContents.add(shelfHeader);
  cabinet.entries.forEach(({item: node, x, y, z}, index) => {
    const group = new THREE.Group();
    group.position.set(x, y, z);
    group.rotation.z = (index % 5 - 2) * .012;
    group.userData.name = `${node.name} · ${node.count} 篇`;
    group.userData.book = true;
    group.userData.category = node.path;
    shelfContents.add(group);
    const categoryFiles = flattenNode(node);
    const updatedAt = Math.max(0, ...categoryFiles.map(file => file.updatedAt ?? 0));
    const age = updatedAt ? Date.now() - updatedAt : Infinity;
    const color = new THREE.Color(bookColors[index % bookColors.length]);
    if (age < 7 * 86400000) color.lerp(new THREE.Color(0xfff5c4), .22);
    else if (age > 90 * 86400000) color.lerp(new THREE.Color(0xc6c5b5), .27);
    const thickness = .48 + Math.min(.55, Math.log2(node.count + 1) * .09);
    // Front cover stays aligned; the paper block grows backwards with content.
    rounded(group, [1.25, 2.3, thickness], [0, 0, (.65 - thickness) / 2], color.getHex(), .11);
    for (let i = 0; i < Math.min(node.children.length, 6); i++) rounded(group, [.055, .12, .035], [-.4 + i * .16, -.95, .36], PALETTE.gold, .01);
    const signal = mesh(cached("beacon", () => new THREE.SphereGeometry(.055, 8, 6)), age < 7 * 86400000 ? beaconFresh : beaconCalm, group, [.43, .96, .39]);
    signal.userData.beacon = true;
    if (categoryFiles.some(file => updateInfo(file).isNew)) {
      rounded(group, [.19, .44, .05], [.35, 1.05, .39], PALETTE.gold, .02);
      const flag = textPlane("NEW", .28, .13, {color:"#5e673b"}); flag.position.set(.35, 1.13, .43); group.add(flag);
    }
    rounded(group, [.055, 2.06, .42], [.625, 0, 0], PALETTE.paper, .01);
    for (const y of [-.95, -.9, .9, .95]) rounded(group, [1.05, .018, .66], [0, y, 0], PALETTE.cream, .004);
    rounded(group, [1.08, .16, .69], [0, .77, .01], PALETTE.gold, .03);
    rounded(group, [1.08, .16, .69], [0, -.77, .01], PALETTE.gold, .03);
    // Long English category names get two lines on the cloth cover.
    const bookName = node.name.length > 8 ? node.name.replace(/[-_]/g, "\n") : node.name;
    const name = textPlane(bookName, 1.08, .75, { weight: 700, color: "#4f4a3f" });
    name.position.set(0, .12, .34);
    group.add(name);
    const count = textPlane(`${node.count} 篇`, .9, .35, { weight: 500, color: "#75694f" });
    count.position.set(0, -.52, .35);
    group.add(count);
    interactive(group, () => { directoryStack.length = 0; openDirectory(node); });
  });

  // Right: a physical update board with suspended cards.
  const board = new THREE.Group();
  board.position.set(7.2, .65, -2.6);
  board.rotation.y = -.08;
  world.add(board);
  rounded(board, [8.1, 7.45, .52], [0, 3.65, -.55], 0xb7a178, .18);
  rounded(board, [7.55, 6.9, .24], [0, 3.65, -.24], 0xece3c9, .12);
  rounded(board, [7.55, 6.9, .16], [0, 3.65, -.88], 0xd8c49e, .12);
  for (const x of [-2.8, 2.8]) rounded(board, [.25, 7, .22], [x, 3.6, -1.03], 0xa48c66, .04);
  const backBrace = rounded(board, [6.6, .23, .2], [0, 3.6, -1.15], 0xb7a178, .04);
  backBrace.rotation.z = .65;
  const updateHeader = textPlane("最近更新 · NEW NOTES", 6.7, .7, { weight: 700, color: "#75654b" });
  updateHeader.position.set(0, 6.85, -.08);
  board.add(updateHeader);
  recentFiles.slice(0, 5).forEach((file, index) => {
    const group = new THREE.Group();
    group.position.set(0, 5.65 - index * 1.14, .02 + index * .025);
    group.rotation.z = (index % 2 ? 1 : -1) * .018;
    board.add(group);
    rounded(group, [6.75, .92, .18], [0, 0, 0], index === 0 ? 0xf3e4b8 : PALETTE.paper, .12);
    sphere(group, [-2.95, 0, .13], [.19, .19, .08], index === 0 ? PALETTE.sage : PALETTE.blush);
    const label = textPlane(`${String(index + 1).padStart(2, "0")}  ${file.title}`, 4.9, .48, { weight: 600, align: "left" });
    label.position.set(-.15, .13, .11);
    group.add(label);
    const info = updateInfo(file);
    const date = textPlane(`${info.bucket} · ${info.relative} · ${file.change === "added" ? "新增" : "修改"}${info.isNew ? " · NEW" : ""}`, 5.7, .28, { weight: 400, align: "left", color: "#8a806d" });
    date.position.set(.15, -.25, .115);
    group.add(date);
    interactive(group, () => window.open(file.url, "_blank", "noopener,noreferrer"));
  });

  // Center desk, mascot, search console and profile frame.
  const desk = new THREE.Group();
  desk.position.set(0, .15, .7);
  world.add(desk);
  rounded(desk, [8.2, .38, 3.4], [0, 1.45, 0], 0xd7c29b, .16);
  rounded(desk, [7.9, .15, 3.2], [0, 1.68, 0], PALETTE.cream, .12);
  for (const x of [-3.35, 3.35]) for (const z of [-1.15, 1.15]) rounded(desk, [.22, 1.45, .22], [x, .7, z], 0xb7a17b, .06);
  for (let i = 0; i < 3; i++) {
    rounded(desk, [1.5, .18, 1.05], [-2.55, 1.9 + i * .22, .3], [PALETTE.sage, PALETTE.blush, PALETTE.gold][i], .05).rotation.y = (i - 1) * .15;
  }
  const bird = new THREE.Group();
  bird.position.set(0, 2.55, -.35);
  desk.add(bird);
  sphere(bird, [0, 0, 0], [.86, .94, .72], PALETTE.cream);
  sphere(bird, [-.68, -.1, .02], [.27, .48, .38], 0xeee2c9).rotation.z = -.38;
  sphere(bird, [.68, -.1, .02], [.27, .48, .38], 0xeee2c9).rotation.z = .38;
  for (const x of [-.27, .27]) {
    sphere(bird, [x, .18, .66], [.07, .09, .04], 0x584c40);
    sphere(bird, [x * 1.35, -.04, .66], [.12, .065, .035], 0xe9beb3);
  }
  const beak = mesh(new THREE.ConeGeometry(.13, .24, 4), toon(PALETTE.gold), bird, [0, .03, .79]);
  beak.rotation.x = Math.PI / 2;
  const bow = new THREE.Group();
  bow.position.set(.52, .72, .2);
  bow.rotation.z = -.22;
  bird.add(bow);
  sphere(bow, [-.18, 0, 0], [.28, .18, .11], PALETTE.sage);
  sphere(bow, [.18, 0, 0], [.28, .18, .11], PALETTE.sage);
  sphere(bow, [0, 0, .05], [.1, .1, .08], PALETTE.gold);
  let birdJumpAt = -Infinity;
  bird.userData.name = "小鸟 · 点击跳一跳";
  interactive(bird, () => { birdJumpAt = performance.now(); lastActivity = birdJumpAt; });

  const searchConsole = new THREE.Group();
  searchConsole.position.set(0, 1.95, 1.95);
  searchConsole.rotation.x = -.2;
  desk.add(searchConsole);
  rounded(searchConsole, [4.8, 1.2, .34], [0, 0, 0], PALETTE.sage, .2);
  const searchLabel = textPlane("⌕  点击搜索全部笔记", 4.25, .72, { weight: 600, background: "#faffee", border: "#d7e2bf" });
  searchLabel.position.z = .2;
  searchConsole.add(searchLabel);
  interactive(searchConsole, () => { directoryStack.length = 0; showSearch(searchQuery); options.onSearchRequest(); });

  const profile = new THREE.Group();
  profile.position.set(3.05, 2.75, .15);
  profile.rotation.y = -.2;
  desk.add(profile);
  rounded(profile, [2.15, 2.5, .28], [0, 0, 0], PALETTE.blush, .16);
  const profileText = textPlane("GOLEMON\nGitHub", 1.8, .85, { weight: 700, color: "#66564b", background: "#fff8ed" });
  profileText.position.z = .16;
  profile.add(profileText);
  interactive(profile, () => window.open(PROFILE_URL, "_blank", "noopener,noreferrer"));
  const csdn = rounded(desk, [1.8, .55, .2], [3.05, 1.98, .35], PALETTE.sage);
  const csdnLabel = textPlane("CSDN", 1.5, .35);
  csdnLabel.position.z = .12;
  csdn.add(csdnLabel);
  interactive(csdn, () => window.open(CSDN_URL, "_blank", "noopener,noreferrer"));

  // Modal notebook exists inside the same WebGL world.
  const panel = new THREE.Group();
  panel.position.set(0, 4.4, 5.2);
  panel.visible = false;
  scene.add(panel);
  let panelTitle = "";
  let panelFiles: DirectoryEntry[] = [];
  let panelPortrait = false;
  let panelScroll = 0;
  let searchQuery = "";
  const directoryStack: BlogTreeNode[] = [];
  function openDirectory(node: BlogTreeNode, push = true) {
    if (push) directoryStack.push(node);
    const entries: DirectoryEntry[] = [
      ...node.children.map(folder => ({title: `▸ ${folder.name} / ${folder.count} 篇`, path: folder.path, url: "", folder})),
      ...node.files,
    ];
    showPanel(node.name, entries, 0);
    announce({view:"category", path:node.path});
  }
  let scrollLimit = 0;
  let listTop = 0;
  let listBottom = 0;
  let scrollRows: { group: THREE.Group; y: number; halfHeight: number }[] = [];
  let scrollThumb: THREE.Mesh | null = null;
  function scrollPanel(value: number) {
    lastActivity = performance.now();
    panelScroll = Math.max(0, Math.min(value, scrollLimit));
    resetHover();
    outline.selectedObjects = [];
    for (const row of scrollRows) {
      row.group.position.y = row.y + panelScroll;
      const record = row.group.userData.hit as Interactive | undefined;
      if (record) record.basePosition.copy(row.group.position);
      row.group.visible = row.group.position.y + row.halfHeight >= listBottom && row.group.position.y - row.halfHeight <= listTop;
    }
    if (scrollThumb) {
      scrollThumb.position.y = listTop - .35 - (scrollLimit ? panelScroll / scrollLimit : 0) * (listTop - listBottom - .7);
      // The thumb casts a shadow, so moving it must refresh the on-demand shadow map.
      renderer.shadowMap.needsUpdate = true;
    }
    renderer.domElement.setAttribute("aria-label", `${panelTitle}，${panelFiles.length} 项，可上下滚动，滚动进度 ${scrollLimit ? Math.round(panelScroll / scrollLimit * 100) : 100}%。Escape 返回。`);
  }
  function clearPanel() {
    resetHover();
    renderer.shadowMap.needsUpdate = true;
    outline.selectedObjects = [];
    for (const record of panelInteractives) {
      const index = interactives.indexOf(record);
      if (index >= 0) interactives.splice(index, 1);
    }
    panelInteractives.clear();
    panel.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      if (!sharedGeometries.has(object.geometry)) object.geometry.dispose();
      const texture = object.userData.textTexture as THREE.Texture | undefined;
      if (texture) {
        texture.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => material.dispose());
      } else if (object.userData.ownedMaterial) {
        object.material.dispose();
      }
    });
    panel.clear();
    scrollRows = [];
    scrollThumb = null;
  }
  function closePanel() {
    if (directoryStack.length > 1) {
      directoryStack.pop();
      openDirectory(directoryStack[directoryStack.length - 1], false);
      return;
    }
    directoryStack.length = 0;
    panel.visible = false;
    controls.enableRotate = true;
    controls.enableZoom = true;
    clearPanel();
    renderer.domElement.setAttribute("aria-label", "三维博客场景，可拖动旋转、滚轮缩放并点击物件");
    if (compact.matches) shelfView(); else homeView();
  }
  function panelButton(parent: THREE.Object3D, label: string, x: number, y: number, action: () => void, width = 1.7) {
    const group = new THREE.Group();
    group.position.set(x, y, .55);
    parent.add(group);
    rounded(group, [width, .68, .22], [0, 0, 0], PALETTE.sageDark, .16);
    const text = textPlane(label, width - .2, .42, { weight: 600, color: "#fffdf5" });
    text.position.z = .13;
    group.add(text);
    interactive(group, action, true);
  }
  function showPanel(titleText: string, files: DirectoryEntry[], scrollOffset: number) {
    clearPanel();
    currentView = "panel";
    controls.enableRotate = false;
    controls.enableZoom = false;
    panelTitle = titleText;
    panelFiles = files;
    panelScroll = scrollOffset;
    panel.visible = true;
    renderer.shadowMap.needsUpdate = true;
    const portrait = compact.matches;
    panelPortrait = portrait;
    panel.position.y = portrait ? 6 : 5;
    rounded(panel, [portrait ? 7.25 : 14.8, portrait ? 10.3 : 8.8, .42], [0, 0, 0], 0xc9b78e, .26);
    if (portrait) {
      rounded(panel, [6.85, 9.85, .25], [0, 0, .32], PALETTE.paper, .18);
    } else {
      const leftPage = rounded(panel, [7.05, 8.15, .25], [-3.56, 0, .32], PALETTE.paper, .18);
      leftPage.rotation.y = .035;
      const rightPage = rounded(panel, [7.05, 8.15, .25], [3.56, 0, .32], PALETTE.cream, .18);
      rightPage.rotation.y = -.035;
      rounded(panel, [.28, 8.05, .48], [0, 0, .34], PALETTE.gold, .06);
    }
    const heading = textPlane(`${titleText} · ${files.length} 项`, portrait ? 5.15 : 6.1, .7, { weight: 700, color: "#61754c" });
    heading.position.set(portrait ? -.45 : -3.55, portrait ? 4.25 : 3.55, .61);
    panel.add(heading);
    const hint = textPlane("上下滑动查阅", portrait ? 1.5 : 3, .36, {color: "#8c7a5b"});
    hint.position.set(portrait ? 2.4 : 4.6, portrait ? 4.2 : 3.5, .61);
    panel.add(hint);
    listTop = portrait ? 3.65 : 3;
    listBottom = portrait ? -3.12 : -2.95;
    const rowHeight = portrait ? 1.08 : .92;
    const rowStep = portrait ? 1.42 : 1.23;
    const columns = portrait ? 1 : 2;
    const firstY = listTop - rowHeight / 2;
    scrollLimit = Math.max(0, (Math.ceil(files.length / columns) - 1) * rowStep + rowHeight - (listTop - listBottom));
    const clipping = [
      new THREE.Plane(new THREE.Vector3(0, -1, 0), panel.position.y + listTop),
      new THREE.Plane(new THREE.Vector3(0, 1, 0), -panel.position.y - listBottom),
    ];
    if (scrollLimit > 0) {
      rounded(panel, [.06, listTop - listBottom, .05], [portrait ? 3.26 : 7.07, (listTop + listBottom) / 2, .59], 0xd5d5bc, .02);
      scrollThumb = rounded(panel, [.13, .7, .08], [portrait ? 3.26 : 7.07, listTop - .35, .64], PALETTE.sageDark, .035);
    }
    if (!files.length) {
      const empty = textPlane("没有找到相关笔记\n试试其他关键词吧", portrait ? 5.8 : 10, 1.5, {color: "#6c765b"});
      empty.position.set(0, .5, .6);
      panel.add(empty);
    }
    files.forEach((file, index) => {
      const column = portrait ? 0 : index % 2 ? 1 : -1;
      const row = Math.floor(index / columns);
      const group = new THREE.Group();
      group.position.set(column * 3.55, firstY - row * rowStep, .56);
      group.userData.scrollRow = true;
      scrollRows.push({group, y: group.position.y, halfHeight: rowHeight / 2});
      panel.add(group);
      rounded(group, [portrait ? 6.15 : 6.25, portrait ? 1.08 : .92, .16], [0, 0, 0], index % 2 ? 0xf7f2e4 : 0xeef3e2, .12);
      const rowText = textPlane(`${index + 1}. ${file.title}`, portrait ? 5.55 : 5.65, .5, { weight: 600, align: "left" });
      rowText.position.z = .1;
      group.add(rowText);
      group.traverse(object => {
        if (!(object instanceof THREE.Mesh)) return;
        const original = object.material;
        object.material = original.clone();
        if (object.userData.textTexture) original.dispose();
        object.material.clippingPlanes = clipping;
        object.userData.ownedMaterial = true;
        object.castShadow = false;
      });
      interactive(group, () => file.folder ? openDirectory(file.folder) : window.open(file.url, "_blank", "noopener,noreferrer"), true);
    });
    panelButton(panel, "返回", portrait ? -2.5 : -5.9, portrait ? -4.45 : -3.55, closePanel, 1.45);
    if (titleText.startsWith("搜索")) panelButton(panel, "输入关键词", portrait ? -.9 : -2.8, portrait ? -3.65 : -3.55, options.onSearchRequest, 2.4);
    scrollPanel(scrollOffset);
    fitView(panel.position.clone(), portrait ? 7.7 : 15.8, portrait ? 11 : 9.8);
  }
  function showSearch(value: string) {
    searchQuery = value;
    directoryStack.length = 0;
    const query = value.trim().toLowerCase();
    const results = searchFiles(value);
    showPanel(query ? `搜索：${value.trim()}` : "搜索全部笔记", results, 0);
    announce({view:"search", q:value});
  }

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let hovered: Interactive | null = null;
  function resetHover() {
    if (hovered) { hovered.root.scale.copy(hovered.baseScale); hovered.root.position.copy(hovered.basePosition); renderer.shadowMap.needsUpdate = true; }
    hovered = null; outline.selectedObjects = []; options.onHover("");
  }
  function highlight(record: Interactive | null) {
    if (record === hovered) return;
    resetHover(); hovered = record;
    if (!record) return;
    record.root.scale.copy(record.baseScale).multiplyScalar(1.035);
    // Pull books out of the cabinet; lift other labels without changing hit areas.
    record.root.position.z = record.basePosition.z + (record.root.userData.book ? .25 : .055);
    renderer.shadowMap.needsUpdate = true;
    outline.selectedObjects = [record.root]; options.onHover(record.name);
  }
  let downAt = new THREE.Vector2();
  let dragged = false;
  let pressed: Interactive | null = null;
  let activePointer: number | null = null;
  let lastPointerY = 0;
  function hitTest(event: PointerEvent) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(panel.visible ? [panel] : [world], true);
    for (const hit of hits) {
      let object: THREE.Object3D | null = hit.object;
      let visible = true;
      while (object) { if (!object.visible) visible = false; object = object.parent; }
      if (!visible) continue;
      let ancestor: THREE.Object3D | null = hit.object;
      let isRow = false;
      while (ancestor) { if (ancestor.userData.scrollRow) isRow = true; ancestor = ancestor.parent; }
      if (isRow) {
        const localY = hit.point.y - panel.position.y;
        if (localY > listTop || localY < listBottom) continue;
      }
      const record = hit.object.userData.hit as Interactive | undefined;
      if (record) return record;
      // Transparent labels should not block the solid object below them.
      if (hit.object instanceof THREE.Mesh && !hit.object.userData.textTexture) return null;
    }
    return null;
  }
  let hoverPointer: PointerEvent | null = null;
  let hoverFrame = 0;
  // Raycasting the whole scene is costly and pointermove can outpace the display; test at most once per frame.
  function hoverHitTest() {
    hoverFrame = 0;
    if (disposed) return;
    const event = hoverPointer;
    if (!event) return;
    hoverPointer = null;
    highlight(hitTest(event));
    renderer.domElement.style.cursor = hovered ? "pointer" : "grab";
  }
  function onPointerMove(event: PointerEvent) {
    lastActivity = performance.now();
    if (downAt.distanceTo(new THREE.Vector2(event.clientX, event.clientY)) > 7) dragged = true;
    if (panel.visible && activePointer === event.pointerId && dragged) {
      const unitsPerPixel = 2 * (camera.position.z - panel.position.z) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / host.clientHeight;
      scrollPanel(panelScroll + (lastPointerY - event.clientY) * unitsPerPixel);
      lastPointerY = event.clientY;
      return;
    }
    // While orbit-dragging the hover target is meaningless; skip the raycast entirely.
    if (dragged && activePointer !== null) { resetHover(); renderer.domElement.style.cursor = "grabbing"; return; }
    hoverPointer = event;
    if (!hoverFrame) hoverFrame = requestAnimationFrame(hoverHitTest);
  }
  function onPointerDown(event: PointerEvent) {
    downAt.set(event.clientX, event.clientY);
    dragged = false;
    pressed = hitTest(event);
    highlight(pressed);
    lastActivity = performance.now();
    activePointer = event.pointerId;
    lastPointerY = event.clientY;
    if (panel.visible) renderer.domElement.setPointerCapture(event.pointerId);
  }
  function onPointerUp(event: PointerEvent) {
    const hit = hitTest(event);
    if (!dragged && hit && hit === pressed) { options.onInteraction(); options.onHover(hit.name); hit.action(); }
    if (dragged) options.onInteraction();
    pressed = null;
    activePointer = null;
    // The orbit-drag branch set "grabbing"; no hover re-test is scheduled without movement, so restore here.
    renderer.domElement.style.cursor = hovered ? "pointer" : "grab";
    if (renderer.domElement.hasPointerCapture(event.pointerId)) renderer.domElement.releasePointerCapture(event.pointerId);
  }
  function onPointerCancel() { activePointer = null; pressed = null; dragged = true; renderer.domElement.style.cursor = "grab"; }
  function onWheel(event: WheelEvent) {
    lastActivity = performance.now(); options.onInteraction();
    if (!panel.visible) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? host.clientHeight : 1);
    scrollPanel(panelScroll + pixels * (listTop - listBottom) / host.clientHeight * 1.5);
  }
  function onKeyDown(event: KeyboardEvent) {
    if (panel.visible && ["ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const step = event.key.startsWith("Page") ? listTop - listBottom : .7;
      scrollPanel(event.key === "Home" ? 0 : event.key === "End" ? scrollLimit : panelScroll + (event.key.endsWith("Down") ? step : -step));
    }
    if (event.key === "Escape") {
      (document.activeElement as HTMLElement | null)?.blur();
      if (panel.visible) closePanel();
    }
  }
  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointerup", onPointerUp);
  renderer.domElement.addEventListener("pointercancel", onPointerCancel);
  renderer.domElement.addEventListener("wheel", onWheel, {passive: false, capture: true});
  window.addEventListener("keydown", onKeyDown);
  let historyTimer: ReturnType<typeof setTimeout>;
  controls.addEventListener("start", () => { controls.autoRotate = false; lastActivity = performance.now(); });
  controls.addEventListener("end", () => {
    clearTimeout(historyTimer);
    historyTimer = setTimeout(() => {
      if (!disposed && !panel.visible && !movingCamera) options.onRouteChange({...currentRoute, camera:[...camera.position.toArray(), ...controls.target.toArray()]}, true);
    }, 400);
  });

  let disposed = false;
  let frame = 0;
  let renderTimeout: ReturnType<typeof setTimeout>;
  const timer = new THREE.Timer();
  timer.connect(document);
  function animate(timestamp?: number) {
    if (disposed || document.hidden) return;
    timer.update(timestamp);
    const t = timer.getElapsed();
    const idleFor = performance.now() - lastActivity;
    const active = movingCamera || idleFor < 1800;
    if (!reduced.matches && !low) {
      board.rotation.z = Math.sin(t * .35) * .006;
      titleBoard.rotation.z = Math.sin(t * .28) * .006;
      const pulse = .48 + Math.sin(t * 1.7) * .2;
      beaconFresh.opacity = pulse; beaconCalm.opacity = pulse;
      const glowPulse = .035 + Math.sin(t * 1.7) * .02;
      for (const glow of glowCache.values()) glow.emissiveIntensity = glowPulse;
    }
    const jumpProgress = (performance.now() - birdJumpAt) / 850;
    const jumpHeight = jumpProgress < 1 ? Math.sin(Math.PI * Math.max(0, jumpProgress)) * (reduced.matches ? .25 : 1.65) : 0;
    bird.position.y = 2.55 + jumpHeight + (!reduced.matches && !low && jumpProgress >= 1 ? Math.sin(t * 1.2) * .08 : 0);
    if (jumpProgress < 1 || jumpProgress < 1.1) renderer.shadowMap.needsUpdate = true;
    if (movingCamera) {
      camera.position.lerp(desiredPosition, .075);
      controls.target.lerp(desiredTarget, .075);
      camera.lookAt(controls.target);
      if (camera.position.distanceTo(desiredPosition) < .04 && controls.target.distanceTo(desiredTarget) < .04) {
        movingCamera = false;
        controls.enabled = true;
      }
    }
    if (!movingCamera) controls.update();
    if (low) renderer.render(scene, camera); else composer.render();
    // No animation work while hidden; idle scenes redraw at 6–12 fps, dropping to 3 fps once deeply idle.
    const fps = active ? (quality === "high" ? 60 : 30) : idleFor > 30000 ? 3 : low ? 6 : 12;
    renderTimeout = setTimeout(() => { frame = requestAnimationFrame(animate); }, Math.max(0, 1000 / fps - 16));
  }
  function visibilityChanged() { clearTimeout(renderTimeout); cancelAnimationFrame(frame); if (!document.hidden) { lastActivity = performance.now(); frame = requestAnimationFrame(animate); } }
  function contextLost(event: Event) { event.preventDefault(); options.onError(); }
  document.addEventListener("visibilitychange", visibilityChanged);
  renderer.domElement.addEventListener("webglcontextlost", contextLost);
  function resize() {
    const width = host.clientWidth;
    const height = host.clientHeight;
    camera.aspect = width / height;
    camera.fov = 38;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(width, height);
    composer.setSize(width, height);
    outline.resolution.set(width, height);
    suppressRoute = true;
    if (currentView === "panel") {
      // Rebuilding every row (canvas textures, cloned materials) is costly; the layout only depends on
      // portrait vs landscape, so plain resizes just refit the camera.
      if (compact.matches !== panelPortrait) showPanel(panelTitle, panelFiles, panelScroll);
      else fitView(panel.position.clone(), panelPortrait ? 7.7 : 15.8, panelPortrait ? 11 : 9.8);
    }
    else if (currentView === "shelf") shelfView();
    else if (currentView === "recent") recentView();
    else homeView();
    suppressRoute = false;
  }
  const resizer = new ResizeObserver(resize);
  resizer.observe(host);
  resize();
  animate();
  const readyFrame = requestAnimationFrame(() => { if (!disposed) options.onReady(); });

  return {
    previewArticle(file: BlogFile) {
      resetHover();
      if (panel.visible) { panel.visible = false; clearPanel(); }
      controls.enableRotate = true; controls.enableZoom = true;
      currentView = "shelf";
      const book = interactives.find(item => item.root.userData.book && (file.path.startsWith(item.root.userData.category + "/") || item.root.userData.category === "root" && !file.path.includes("/")));
      if (book) {
        world.updateMatrixWorld(true);
        const center = book.root.getWorldPosition(new THREE.Vector3());
        fitView(center, 6, 6, .08);
        highlight(book);
      }
    },
    navigate(route: Route) {
      suppressRoute = true;
      resetHover();
      if (panel.visible) { panel.visible = false; clearPanel(); }
      controls.enableRotate = true; controls.enableZoom = true;
      directoryStack.length = 0;
      if (route.view === "category") {
        const node = nodes.find(n => n.path === route.path);
        if (node) {
          directoryStack.push(...nodes.filter(n => node.path.startsWith(n.path + "/")).sort((a,b) => a.path.length - b.path.length));
          openDirectory(node);
        } else shelfView();
      } else if (route.view === "search") showSearch(route.q ?? "");
      else if (route.view === "recent") recentView();
      else if (route.view === "shelf") shelfView();
      else homeView();
      if (route.camera && !panel.visible) {
        const position = new THREE.Vector3(...route.camera.slice(0,3) as [number,number,number]);
        const target = new THREE.Vector3(...route.camera.slice(3) as [number,number,number]);
        // Only the full-room view is centred on the room; focused views retain their own targets.
        if (Math.abs(target.x) > 1 && route.view === "home") {
          const offset = position.clone().sub(target);
          offset.setLength(Math.max(offset.length(), desiredPosition.distanceTo(desiredTarget)));
          goTo(overviewTarget.clone().add(offset), overviewTarget.clone());
        } else if (position.distanceTo(target) >= 5) goTo(position, target);
      }
      currentRoute = route;
      suppressRoute = false;
    },
    setQuery(value: string) { showSearch(value); },
    dispose() {
      disposed = true;
      clearTimeout(renderTimeout); clearTimeout(historyTimer);
      cancelAnimationFrame(frame);
      cancelAnimationFrame(readyFrame);
      cancelAnimationFrame(hoverFrame);
      resizer.disconnect();
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointercancel", onPointerCancel);
      renderer.domElement.removeEventListener("wheel", onWheel, true);
      window.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("visibilitychange", visibilityChanged);
      renderer.domElement.removeEventListener("webglcontextlost", contextLost);
      controls.dispose();
      timer.dispose();
      scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        object.geometry.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        for (const material of materials) {
          if (material.map) material.map.dispose();
          material.dispose();
        }
      });
      gradient.dispose();
      composer.passes.forEach(pass => pass.dispose());
      composer.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
