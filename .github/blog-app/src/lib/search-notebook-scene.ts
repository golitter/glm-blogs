import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

export function createSearchNotebook(host: HTMLElement) {
  const renderer = new THREE.WebGLRenderer({alpha:true, antialias:true, powerPreference:"low-power"});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x879771, 2));
  const lamp = new THREE.DirectionalLight(0xffefd5, 2.1); lamp.position.set(-4,6,10); scene.add(lamp);
  const camera = new THREE.OrthographicCamera(-4,4,4,-4,.1,40); camera.position.set(0,0,12);
  const book = new THREE.Group(); scene.add(book);
  const cream = new THREE.MeshStandardMaterial({color:0xfff7e2, roughness:.87});
  const paper = new THREE.MeshStandardMaterial({color:0xfffbea, roughness:1});
  const sage = new THREE.MeshStandardMaterial({color:0x809664, roughness:.64});
  const gold = new THREE.MeshStandardMaterial({color:0xe4c47e, metalness:.7, roughness:.25});
  const materials = [cream,paper,sage,gold];
  function box(w: number,h: number,d: number,x: number,y: number,z: number,material: THREE.Material,r=.08) {
    const mesh = new THREE.Mesh(new RoundedBoxGeometry(w,h,d,3,r),material);
    mesh.position.set(x,y,z); book.add(mesh); return mesh;
  }
  let disposed = false, frame = 0, lastSize = "";
  function resize() {
    if (disposed) return;
    const width = host.clientWidth, height = host.clientHeight;
    // Rebuilding all geometry is wasteful; skip when the measured size has not actually changed.
    const size = `${width}x${height}`;
    if (!width || !height || size === lastSize) return;
    lastSize = size;
    book.traverse(obj => {if (obj instanceof THREE.Mesh) obj.geometry.dispose();}); book.clear();
    renderer.setSize(width,height);
    camera.left = -width/200; camera.right = width/200; camera.top = height/200; camera.bottom = -height/200; camera.updateProjectionMatrix();
    const w = width/100 - .32, h = height/100 - .35;
    box(w,h,.25,0,-.02,-.2,sage,.15);
    // Offset layers expose the paper edges along the right and bottom of the cover.
    for(let i=0;i<5;i++) box(w-.22,h-.2,.035,.025+i*.008,.02+i*.008,-.04+i*.037,i%2 ? cream : paper,.1);
    box(w-.32,h-.28,.08,0,.05,.19,paper,.12);
    box(.18,h-.12,.31,-w/2+.2,0,.05,sage,.05);
    // Metal rings have actual curved geometry and highlights, not CSS shadows.
    for(const x of [-.28,.28]) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(.14,.037,10,32),gold);
      ring.position.set(x,h/2-.23,.38); ring.rotation.x = -.65; book.add(ring);
      box(.14,.08,.04,x,h/2-.37,.27,sage,.02);
    }
    box(.38,.7,.09,w/2-.43,-h/2+.32,.28,sage,.03);
    box(.42,.08,.11,w/2-.43,-h/2+.64,.31,gold,.02);
    book.rotation.set(-.022,-.035,-.005);
    renderer.render(scene,camera);
  }
  function move(event: PointerEvent) {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const rect = host.getBoundingClientRect();
    book.rotation.y = -.035 + ((event.clientX-rect.left)/rect.width-.5)*.025;
    book.rotation.x = -.022 + ((event.clientY-rect.top)/rect.height-.5)*.015;
    if (!frame) frame = requestAnimationFrame(() => {frame=0; if(!disposed) renderer.render(scene,camera);});
  }
  const observer = new ResizeObserver(resize); observer.observe(host); resize();
  const parent = host.parentElement!; parent.addEventListener("pointermove",move);
  return () => {disposed=true; cancelAnimationFrame(frame); observer.disconnect(); parent.removeEventListener("pointermove",move); book.traverse(obj => {if(obj instanceof THREE.Mesh) obj.geometry.dispose();}); materials.forEach(m=>m.dispose()); renderer.dispose(); renderer.domElement.remove();};
}
