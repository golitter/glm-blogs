// Dev-only probe: drives the page over CDP and reports frame gaps (rAF deltas) around key interactions.
// Usage: node scripts/measure-longtasks.mjs [url] (expects a CDP endpoint on 127.0.0.1:9222)
const url = process.argv[2] ?? "http://127.0.0.1:4173/";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const list = await (await fetch("http://127.0.0.1:9222/json/list")).json();
const target = list.find((t) => t.type === "page");
if (!target) throw new Error("no page target");
const ws = new WebSocket(target.webSocketDebuggerUrl);
let seq = 0;
const pending = new Map();
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
}
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
  }
};
await new Promise((r) => (ws.onopen = r));
await send("Page.enable");
await send("Runtime.enable");
// Simulate a mid-range mobile CPU so sub-50ms work becomes visible as frame drops.
const throttle = Number(process.argv[3] ?? 4);
if (throttle > 1) await send("Emulation.setCPUThrottlingRate", { rate: throttle });
console.log(`CPU 节流 ${throttle}×`);
async function evaluate(expression) {
  const { result, exceptionDetails } = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (exceptionDetails) throw new Error(exceptionDetails.text + " " + (exceptionDetails.exception?.description ?? ""));
  return result.value;
}
function report(label, gaps) {
  const sorted = [...gaps].sort((a, b) => b - a);
  const sum = gaps.reduce((s, g) => s + g, 0);
  console.log(`${label}: 帧数 ${gaps.length}, 平均帧距 ${(sum / Math.max(1, gaps.length)).toFixed(1)}ms, 最差 ${sorted[0] ?? 0}ms, 前5差 ${sorted.slice(0, 5).map(g => Math.round(g)).join("/")}ms, >100ms 帧数 ${gaps.filter(g => g > 100).length}`);
}
await send("Page.navigate", { url });
await sleep(300);
await evaluate(`new Promise(done => {
  window.__gaps = []; window.__last = 0;
  window.__sample = () => { const now = performance.now(); if (window.__last) window.__gaps.push(now - window.__last); window.__last = now; requestAnimationFrame(window.__sample); };
  requestAnimationFrame(window.__sample);
  const start = performance.now();
  new MutationObserver((_, obs) => {
    if (document.querySelector('.webgl-world canvas') && performance.now() - start > 100) { obs.disconnect(); done(true); }
  }).observe(document.body, {childList: true, subtree: true});
  setTimeout(() => done(false), 30000);
})`);
await sleep(2500);
report("启动(含场景构建)", await evaluate(`window.__gaps.splice(0)`));

await evaluate(`location.hash = '#/search'`);
await sleep(3000);
console.log("面板状态:", await evaluate(`document.querySelector('.webgl-world canvas')?.getAttribute('aria-label')?.slice(0, 60)`));
report("打开全量搜索面板", await evaluate(`window.__gaps.splice(0)`));

await evaluate(`new Promise(done => {
  const canvas = document.querySelector('.webgl-world canvas');
  let round = 0;
  const timer = setInterval(() => {
    for (let i = 0; i < 12; i++) canvas.dispatchEvent(new WheelEvent('wheel', {deltaY: 900, bubbles: true, cancelable: true}));
    if (++round >= 30) { clearInterval(timer); setTimeout(done, 400); }
  }, 40);
})`);
report("滚动构建全部行", await evaluate(`window.__gaps.splice(0)`));

await evaluate(`location.hash = '#/'`);
await sleep(1500);
await evaluate(`window.__gaps.splice(0)`);
await evaluate(`new Promise(done => {
  const canvas = document.querySelector('.webgl-world canvas');
  const rect = canvas.getBoundingClientRect();
  for (let i = 0; i < 80; i++) {
    canvas.dispatchEvent(new PointerEvent('pointermove', {clientX: rect.left + rect.width * (0.18 + (i % 12) * 0.02), clientY: rect.top + rect.height * (0.32 + (i % 9) * 0.035), pointerId: 1, bubbles: true, isPrimary: true}));
  }
  setTimeout(done, 1500);
})`);
report("悬停扫描书架", await evaluate(`window.__gaps.splice(0)`));
ws.close();
