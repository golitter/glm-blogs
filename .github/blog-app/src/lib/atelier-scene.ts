import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export function createAtelier(host: HTMLElement, onReady: () => void, onLost: () => void) {
  const compact = matchMedia("(max-width: 640px)");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, compact.matches ? 1.25 : 1.75));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.setClearColor(0xfffdf6, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.3;
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0xfffdf6, 0.018);
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(7.8, 5.8, 11.8);
  camera.lookAt(0, 1.05, 0);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1.05, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.minPolarAngle = Math.PI * 0.24;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.minAzimuthAngle = -0.72;
  controls.maxAzimuthAngle = 0.72;
  controls.autoRotate = !reduced.matches;
  controls.autoRotateSpeed = 0.3;
  controls.update();
  scene.add(new THREE.HemisphereLight(0xfff9ec, 0x7f8c69, 2.15));
  const sun = new THREE.DirectionalLight(0xfff1d3, 5.2);
  sun.position.set(-4, 9, 7);
  sun.castShadow = true;
  sun.shadow.mapSize.set(compact.matches ? 512 : 1024, compact.matches ? 512 : 1024);
  Object.assign(sun.shadow.camera, { left: -5, right: 5, top: 5, bottom: -5 });
  sun.shadow.normalBias = 0.045;
  sun.shadow.bias = -0.0003;
  sun.shadow.radius = 4;
  scene.add(sun);
  const world = new THREE.Group();
  scene.add(world);
  const materials = new Map<string, THREE.MeshPhysicalMaterial>();
  function material(color: string) {
    if (!materials.has(color)) {
      const isGold = color === "#d6b770";
      const isPorcelain = color === "#fff8e9" || color === "#fffdf7";
      materials.set(color, new THREE.MeshPhysicalMaterial({
        color,
        roughness: isGold ? 0.24 : isPorcelain ? 0.32 : 0.58,
        metalness: isGold ? 0.72 : 0,
        clearcoat: isPorcelain ? 0.65 : 0.18,
        clearcoatRoughness: 0.25,
      }));
    }
    return materials.get(color)!;
  }
  function mesh(geometry: THREE.BufferGeometry, color: string, position: [number, number, number], parent: THREE.Object3D = world) {
    const item = new THREE.Mesh(geometry, material(color));
    item.position.set(...position); item.castShadow = true; item.receiveShadow = true; parent.add(item); return item;
  }
  function ball(color: string, position: [number, number, number], scale: [number, number, number], parent: THREE.Object3D = world) {
    const item = mesh(new THREE.SphereGeometry(1, compact.matches ? 20 : 32, 20), color, position, parent);
    item.scale.set(...scale); return item;
  }
  function box(color: string, position: [number, number, number], size: [number, number, number], parent: THREE.Object3D = world) {
    return mesh(new RoundedBoxGeometry(...size, 3, Math.min(...size) * 0.2), color, position, parent);
  }
  const cream = "#fff8e9", green = "#bfd39e", gold = "#d6b770";
  // A deep, physical stage: rear halo, floating rings and floor provide clear parallax.
  const haloMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xe8f0d8,
    roughness: 0.2,
    transmission: 0.18,
    transparent: true,
    opacity: 0.72,
    side: THREE.DoubleSide,
  });
  const halo = new THREE.Mesh(new THREE.CircleGeometry(3.35, 64), haloMaterial);
  halo.position.set(0, 1.25, -2.65);
  halo.receiveShadow = true;
  world.add(halo);
  const haloRing = mesh(new THREE.TorusGeometry(3.48, 0.045, 12, 96), gold, [0, 1.25, -2.55]);
  haloRing.rotation.z = -0.08;
  const rearRing = mesh(new THREE.TorusGeometry(2.55, 0.025, 10, 80), "#fff8e9", [0.45, 1.7, -2.25]);
  rearRing.rotation.z = 0.3;
  // Raised oval platform and cloud cushions form the floating stage.
  const base = mesh(new THREE.CylinderGeometry(2.85, 2.7, 0.28, 64), "#d4dcb5", [0, -0.15, 0]);
  base.scale.z = 0.77;
  const surface = mesh(new THREE.CylinderGeometry(2.8, 2.8, 0.07, 64), cream, [0, 0.025, 0]);
  surface.scale.z = 0.77;
  for (const [x,y,z,s] of [[-2.2,-.35,1,.65],[-1.5,-.45,1.65,.65],[-.55,-.5,1.95,.7],[.55,-.48,1.95,.65],[1.5,-.4,1.55,.75],[2.35,-.25,.75,.6],[-2.7,-.2,-.15,.5]]) {
    ball("#fffdf7", [x,y,z], [s,s*.62,s*.72]);
  }
  box("#ddc9a5", [0, .68, -.2], [3.7,.2,1.8]);
  box(cream, [0,.8,-.2], [3.78,.09,1.87]);
  for (const x of [-1.5,1.5]) for (const z of [-.8,.4]) box("#c4b395", [x,.34,z], [.13,.62,.13]);
  for (let i=0;i<3;i++) {
    const book = new THREE.Group(); world.add(book);
    book.position.set(1.07,.92+i*.19,-.25); book.rotation.y = -.15+i*.14;
    box([green,"#ebd1c6",gold][i], [0,0,0], [.86,.18,.7], book);
    box("#fffbee", [0,.015,.03], [.78,.1,.65], book);
  }
  // Open notebook with dimensional pages, ruled lines and ribbon bookmark.
  const notebook = new THREE.Group(); world.add(notebook); notebook.position.set(-.37,.94,.29); notebook.rotation.y = -.12;
  box(green,[0,-.05,0],[1.65,.07,1.04],notebook);
  for (const side of [-1,1]) {
    const page = box(cream,[side*.4,0,0],[.79,.1,.96],notebook); page.rotation.z = side*-.045;
    for (let i=0;i<5;i++) box("#dedbc6",[side*.4,.065,-.27+i*.13],[.51,.008,.009],notebook);
  }
  box(gold,[.07,.07,.3],[.055,.012,.76],notebook);
  // Kotori-inspired mascot: cream porcelain, warm eyes and a sage bow.
  const bird = new THREE.Group(); world.add(bird); bird.position.set(-.78,1.25,-.38); bird.rotation.y = .28;
  ball(cream,[0,.24,0],[.6,.65,.5],bird);
  ball("#f2e9d6",[-.47,.09,.04],[.19,.36,.33],bird).rotation.z = -.35;
  ball("#f2e9d6",[.47,.09,.04],[.19,.36,.33],bird).rotation.z = .35;
  for (const x of [-.2,.2]) {
    ball("#594f42",[x,.4,.461],[.045,.057,.032],bird);
    ball("#ffffff",[x-.012,.419,.489],[.012,.016,.008],bird);
    ball("#ecc4b3",[x*1.5,.25,.428],[.085,.045,.014],bird);
    ball(gold,[x,-.32,.17],[.13,.07,.17],bird);
  }
  const beak = mesh(new THREE.ConeGeometry(.09,.16,4),gold,[0,.28,.54],bird); beak.rotation.x = Math.PI/2;
  const tuft = ball("#c6b395",[.04,.88,.01],[.13,.32,.1],bird); tuft.rotation.z=-.35;
  const bow = new THREE.Group(); bow.position.set(.34,.78,.16); bow.rotation.z=-.2; bird.add(bow);
  for (const side of [-1,1]) {
    const loop=ball(green,[side*.16,0,0],[.22,.15,.095],bow); loop.rotation.z=side*.35;
    const tail=box(green,[side*.11,-.2,-.01],[.12,.32,.045],bow); tail.rotation.z=side*.32;
  }
  ball(gold,[0,0,.05],[.08,.09,.07],bow);
  const arch=mesh(new THREE.TorusGeometry(1.6,.045,12,80,Math.PI),gold,[0,1.15,-1.03]);
  arch.rotation.y = .02;
  for (const x of [-1.6,1.6]) box(gold,[x,.65,-1.03],[.085,1,.085]);
  mesh(new THREE.CylinderGeometry(.16,.13,.37,24),"#e6c9bb",[1.3,1.05,.43]);
  const stem = box(gold,[1.3,1.53,.43],[.025,.8,.025]); stem.rotation.z=-.22;
  const feather = ball(cream,[1.39,1.78,.43],[.13,.35,.035]); feather.rotation.z=-.35;
  const stars: THREE.Mesh[]=[];
  function star(x:number,y:number,z:number,size:number) {
    const shape=new THREE.Shape();
    for(let i=0;i<8;i++){ const a=i*Math.PI/4; const r=i%2 ? size*.28 : size; const px=Math.sin(a)*r, py=Math.cos(a)*r; if(i===0) shape.moveTo(px,py); else shape.lineTo(px,py); }
    shape.closePath();
    const item=mesh(new THREE.ExtrudeGeometry(shape,{depth:.045,bevelEnabled:true,bevelSize:.015,bevelThickness:.015,bevelSegments:2,steps:1}),gold,[x,y,z]);
    stars.push(item);
  }
  star(-2.3,2.35,-.4,.19); star(2.15,2.6,-.6,.25); star(1.45,3.18,-.8,.12);
  ball(cream,[-2.25,1.3,-1.1],[.36,.21,.23]); ball(cream,[-2.5,1.23,-1],[.25,.17,.2]);
  ball(cream,[2.1,1.88,-.7],[.31,.19,.2]); ball(cream,[2.36,1.85,-.7],[.24,.13,.17]);
  const floorMaterial = new THREE.MeshPhysicalMaterial({ color: 0xf5f0df, roughness: 0.72, transparent: true, opacity: 0.82 });
  const floor = new THREE.Mesh(new THREE.CircleGeometry(7.5, 72), floorMaterial);
  floor.rotation.x=-Math.PI/2; floor.position.y=-.88; floor.receiveShadow=true; scene.add(floor);
  const shadowPlane = new THREE.Mesh(new THREE.PlaneGeometry(20,20),new THREE.ShadowMaterial({opacity:.16}));
  shadowPlane.rotation.x=-Math.PI/2; shadowPlane.position.y=-.86; shadowPlane.receiveShadow=true; scene.add(shadowPlane);

  let visible=false, lost=false, disposed=false, frame=0, first=true;
  const starHeights=stars.map(s=>s.position.y);
  const start=performance.now();
  function draw(now:number) {
    frame=0;
    if(disposed || lost || !visible || document.hidden) return;
    const moving=!reduced.matches;
    const t=(now-start)/1000;
    world.position.y=moving ? Math.sin(t*.65)*.055 : 0;
    controls.update();
    stars.forEach((s,i)=>{s.position.y=starHeights[i]+(moving ? Math.sin(t*.8+i)*.08 : 0);});
    renderer.render(scene,camera);
    if(first){ first=false; onReady(); }
    if(moving) frame=requestAnimationFrame(draw);
  }
  function sync() {
    cancelAnimationFrame(frame); frame=0;
    if(!disposed && !lost && visible && !document.hidden) frame=requestAnimationFrame(draw);
  }
  function resize() {
    const {width,height}=host.getBoundingClientRect();
    if(!width || !height) return;
    camera.aspect=width/height;
    camera.fov=compact.matches ? 39 : 32;
    camera.updateProjectionMatrix(); renderer.setSize(width,height); sync();
  }
  function contextLost(event:Event){event.preventDefault();lost=true;cancelAnimationFrame(frame);onLost();}
  function contextRestored(){lost=false;first=true;resize();}
  const intersection=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;sync();}); intersection.observe(host);
  const resizer=new ResizeObserver(resize); resizer.observe(host);
  const renderOnControl = () => renderer.render(scene, camera);
  controls.addEventListener("change", renderOnControl);
  renderer.domElement.addEventListener("webglcontextlost",contextLost);renderer.domElement.addEventListener("webglcontextrestored",contextRestored);
  document.addEventListener("visibilitychange",sync); reduced.addEventListener("change",sync);
  resize();
  return () => {
    disposed=true;cancelAnimationFrame(frame);intersection.disconnect();resizer.disconnect();
    controls.removeEventListener("change", renderOnControl);
    document.removeEventListener("visibilitychange",sync);reduced.removeEventListener("change",sync);
    renderer.domElement.removeEventListener("webglcontextlost",contextLost);renderer.domElement.removeEventListener("webglcontextrestored",contextRestored);
    controls.dispose();
    scene.traverse(object=>{if(object instanceof THREE.Mesh) object.geometry.dispose();});
    materials.forEach(m=>m.dispose());haloMaterial.dispose();floorMaterial.dispose();shadowPlane.material.dispose();renderer.dispose();renderer.domElement.remove();
  };
}
