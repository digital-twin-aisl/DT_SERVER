// Run against an isolated topic/port when injecting test snapshots.
import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
const browser = await chromium.launch({headless:true,args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport:{width:1200,height:800}});
const errors=[];
page.on('pageerror', e=>errors.push(e.message));
page.on('console', message => { if(message.type()==='error') errors.push(message.text()); });
await page.goto(process.env.VIEWER_URL || 'http://127.0.0.1:8005/viewer');
await page.waitForFunction(()=>document.body.dataset.mapReady==='true',{},{timeout:120000});
await page.screenshot({path:'/tmp/dt-viewer.png',timeout:120000});
console.log(JSON.stringify({mapReady:await page.locator('body').getAttribute('data-map-ready'),status:await page.locator('#connection').textContent(),fps:await page.locator('#fps').textContent(),errors}));
if (process.env.SCENE_SMOKE === '1') {
  writeFileSync('/tmp/dt-smoke-ready', 'ready');
  await page.waitForFunction(()=>document.getElementById('people-count').textContent==='2',{},{timeout:30000});
  assert.equal(await page.locator('#pose-count').textContent(),'1');
  await page.screenshot({path:'/tmp/dt-viewer-people.png',timeout:120000});
  writeFileSync('/tmp/dt-smoke-captured', 'captured');
  await page.waitForFunction(()=>document.getElementById('people-count').textContent==='0',{},{timeout:30000});
  console.log('scene smoke: 2 people, 1 skeleton, empty snapshot removal passed');
}
assert.deepEqual(errors,[]);
await browser.close();
