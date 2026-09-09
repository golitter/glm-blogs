// Dev-only helper: drives a headless browser over CDP and reports the page's runtime memory.
// Usage: node scripts/measure-memory.mjs [url] (expects a CDP endpoint on 127.0.0.1:9222)
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
await send("Performance.enable");
await send("Page.navigate", { url });
const errors = [];
ws.addEventListener("message", (e) => {
  const msg = JSON.parse(e.data);
  if (msg.method === "Runtime.exceptionThrown") errors.push(msg.params.exceptionDetails.text);
});
async function snapshot(label) {
  const { metrics } = await send("Performance.getMetrics");
  const pick = (name) => metrics.find((m) => m.name === name)?.value ?? 0;
  const { result } = await send("Runtime.evaluate", {
    expression: `JSON.stringify({
      perfMemory: performance.memory ? {usedMB: Math.round(performance.memory.usedJSHeapSize/1048576), totalMB: Math.round(performance.memory.totalJSHeapSize/1048576)} : null,
      hasCanvas: !!document.querySelector('.webgl-world canvas'),
      loadingGone: !document.querySelector('.loading-screen'),
      simpleMode: !!document.querySelector('.static-directory'),
      webgl: (() => { try { const c = document.createElement('canvas'); return !!c.getContext('webgl2'); } catch { return false; } })(),
    })`,
    returnByValue: true,
  });
  const info = JSON.parse(result.value);
  console.log(`[${label}] JS堆已用 ${(pick("JSHeapUsedSize") / 1048576).toFixed(1)} MB / 已申请 ${(pick("JSHeapTotalSize") / 1048576).toFixed(1)} MB | DOM节点 ${pick("Nodes")} | 监听器 ${pick("JSEventListeners")} | 画布:${info.hasCanvas} 场景就绪:${info.loadingGone} 简洁模式:${info.simpleMode} WebGL2:${info.webgl}`);
  return info;
}
await snapshot("刚加载(2s)");
await sleep(10000);
await snapshot("场景就绪后(12s)");
// 打开一次搜索面板再量一次
await send("Page.navigate", { url: url + "#/search?q=agent" });
await sleep(6000);
await snapshot("打开搜索面板后");
if (errors.length) console.log("页面异常:", errors.slice(0, 3));
ws.close();
