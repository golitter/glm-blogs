import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
const source = await readFile(new URL('../src/lib/blog-experience.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.ESNext, target:ts.ScriptTarget.ES2022}}).outputText
  .replace('"pinyin-pro"', JSON.stringify(import.meta.resolve('pinyin-pro')))
  .replace('"../generated/blog-data"', JSON.stringify(new URL('../src/generated/blog-data.ts', import.meta.url).href));
const lib = await import('data:text/javascript;base64,' + Buffer.from(compiled).toString('base64'));
test('routes round trip Chinese, nested paths, reserved query characters and camera', () => {
  for (const route of [{view:'category',path:'llm/agent'}, {view:'category',path:'学习/含 空格'}, {view:'search',q:'codex & C++/#'}, {view:'recent'}, {view:'home',camera:[1,2,3,4,5,6]}]) {
    assert.deepEqual(lib.parseRoute(lib.routeHash(route)), route);
  }
  assert.deepEqual(lib.parseRoute('#/category/%broken'), {view:'home'});
  assert.equal(lib.parseRoute('#/recent?camera=NaN,2,3,4,5,6').camera, undefined);
});
test('search supports case folding, pinyin, initials, multiple terms and empty query', () => {
  assert.equal(lib.searchFiles('').length, lib.files.length);
  assert.deepEqual(lib.searchFiles('CODEX'), lib.searchFiles('codex'));
  assert.ok(lib.searchFiles('ceshi').some(f => f.title.includes('测试')));
  assert.ok(lib.searchFiles('hdcs').some(f => f.title.includes('后端测试')));
  assert.ok(lib.searchFiles('backend ceshi').every(f => f.path.includes('backend')));
  assert.equal(lib.searchFiles('___no_match_928375___').length, 0);
});
test('highlight maps pinyin to the matching Chinese characters', () => {
  assert.equal(lib.highlightedParts('后端测试分层', 'ceshi').filter(p => p.hit).map(p => p.text).join(''), '测试');
});
test('NEW is limited to recently added articles, never recent edits', () => {
  const now = Date.UTC(2026,8,9,12), base = {title:'x',path:'x',url:'https://example.com'};
  assert.ok(lib.updateInfo({...base,updatedAt:now-1000,change:'added'}, now).isNew);
  assert.equal(lib.updateInfo({...base,updatedAt:now-1000,change:'modified'}, now).isNew, false);
  assert.equal(lib.updateInfo({...base,updatedAt:now-8*86400000,change:'added'}, now).isNew, false);
  assert.equal(lib.updateInfo(base,now).bucket, '时间未知');
});
test('no-script index contains every actual article URL, safely encoded', async () => {
  const html = await readFile(new URL('../public/directory.html', import.meta.url),'utf8');
  for (const file of lib.files) assert.ok(html.includes(file.url.replaceAll('&','&amp;')));
});
