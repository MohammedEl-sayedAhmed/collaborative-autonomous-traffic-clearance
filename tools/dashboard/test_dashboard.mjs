// Playwright regression test for the dashboard: run selection, guide, interactive charts
// (hover/zoom/box-zoom/back/reset), maximize, outcomes hover, and responsive layout.
//   ./run.sh dashboard-demo && ./run.sh dashboard &     # serve with demo data
//   npm i playwright && npx playwright install chromium
//   DASH_URL=http://127.0.0.1:8770 node tools/dashboard/test_dashboard.mjs

import { chromium } from 'playwright';

const URL = process.env.DASH_URL || 'http://127.0.0.1:8770';
let failures = 0;
const ok = (cond, msg) => { console.log((cond ? '  PASS ' : '  FAIL ') + msg); if (!cond) failures++; };
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const rows = (page) => page.$$eval('.runrow', els => els.map(el => ({
  run: el.getAttribute('data-run'),
  label: (el.querySelector('.lbl').firstChild ? el.querySelector('.lbl').firstChild.textContent : '').trim(),
  checked: el.querySelector('.chk').checked,
  sel: el.classList.contains('sel'),
})));
const tableLabels = (page) => page.$$eval('#cmpTable tbody tr td:first-child', tds => tds.map(td => td.textContent.trim()));
const legendCount = (page) => page.$$eval('#legReward > span', s => s.length);

async function selectionOf(page) {
  const r = await rows(page);
  return {
    checked: r.filter(x => x.checked).map(x => x.label).sort(),
    selClass: r.filter(x => x.sel).map(x => x.label).sort(),
    table: (await tableLabels(page)).sort(),
    all: r,
  };
}

// Assert checkbox state, .sel highlight, and comparison-table membership ALL agree.
async function invariant(page, expected, tag) {
  const s = await selectionOf(page);
  const exp = [...expected].sort();
  ok(eq(s.checked, exp), `${tag}: checkboxes = [${s.checked}] (expected [${exp}])`);
  ok(eq(s.selClass, exp), `${tag}: .sel highlight = [${s.selClass}]`);
  ok(eq(s.table, exp), `${tag}: comparison table = [${s.table}]`);
}

const consoleErrors = [];

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on('pageerror', e => consoleErrors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') consoleErrors.push('console.error: ' + m.text()); });

  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.waitForSelector('.runrow', { timeout: 10000 });
  await page.click('#pauseChk'); // stop the 2s auto-poll so the test is deterministic

  const r0 = await rows(page);
  console.log(`\nFound ${r0.length} runs: ${r0.map(x => x.label).join(', ')}`);
  ok(r0.length >= 2, 'at least 2 runs present');
  const A = r0[0], B = r0[1];
  const click = async (run) => { await page.click(`.runrow[data-run="${run}"]`); await page.waitForTimeout(120); };

  console.log('\n[1] none -> empty');
  await page.click('#selNone'); await page.waitForTimeout(120);
  await invariant(page, [], 'after none');

  console.log('\n[2] select A');
  await click(A.run); await invariant(page, [A.label], 'A on');

  console.log('\n[3] toggle A off');
  await click(A.run); await invariant(page, [], 'A off');

  console.log('\n[4] A on, B on');
  await click(A.run); await click(B.run); await invariant(page, [A.label, B.label], 'A+B');

  console.log('\n[5] A off (B stays)');
  await click(A.run); await invariant(page, [B.label], 'only B');

  console.log('\n[6] rapid 5x toggle A (odd -> ends ON)');
  for (let i = 0; i < 5; i++) await click(A.run);
  await invariant(page, [A.label, B.label], 'A rapid5 -> on');

  console.log('\n[7] rapid 4x toggle B (even -> stays ON)');
  for (let i = 0; i < 4; i++) await click(B.run);
  await invariant(page, [A.label, B.label], 'B rapid4 -> on');

  console.log('\n[8] interleaved A,B,A,B,B');
  await page.click('#selNone'); await page.waitForTimeout(120);
  await click(A.run); await click(B.run); await click(A.run); await click(B.run); await click(B.run);
  // A: on,off => off ; B: on,off,on => on
  await invariant(page, [B.label], 'interleaved');

  console.log('\n[9] all');
  await page.click('#selAll'); await page.waitForTimeout(120);
  await invariant(page, r0.map(x => x.label), 'all');

  console.log('\n[T] tiles stay consistent with selection (no lag after data loads)');
  await page.waitForTimeout(600);
  const tilesTxt = await page.$eval('#tiles', el => el.textContent);
  ok(/Runs compared|Total episodes/.test(tilesTxt), 'tiles show multi-run summary when 2 runs selected');

  console.log('\n[10] none again');
  await page.click('#selNone'); await page.waitForTimeout(120);
  await invariant(page, [], 'none again');

  console.log('\n[11] legend follows selection (after async fetch)');
  await click(A.run); await click(B.run);
  await page.waitForTimeout(600); // allow pollSelected fetch + redraw
  const lc = await legendCount(page);
  ok(lc === 2, `reward legend shows ${lc} series (expected 2)`);

  console.log('\n[G] guide panel opens/closes and explains the terms');
  await page.click('#guideBtn'); await page.waitForTimeout(150);
  ok(await page.isVisible('#guide .sheet'), 'guide opens on button click');
  const gtext = await page.$eval('#guide .sheet', el => el.textContent);
  ok(gtext.includes('Q-table') && gtext.includes('Episode') && gtext.includes('Epsilon'), 'guide explains RL/Q-table/episode/epsilon');
  await page.click('#guideClose'); await page.waitForTimeout(150);
  ok(!(await page.isVisible('#guide .sheet')), 'guide closes on × click');

  console.log('\n[L] all four charts fit on one row at 2K width (single chart row)');
  await page.setViewportSize({ width: 2560, height: 1440 });
  await page.waitForTimeout(200);
  const chartTops = await page.$$eval('.grid2 .card', els => els.map(e => Math.round(e.getBoundingClientRect().top)));
  const distinctRows = new Set(chartTops).size;
  ok(distinctRows === 1, `4 charts share one row at 2560px (distinct top offsets: ${distinctRows})`);

  console.log('\n[I] interactive charts: hover tooltip, scroll-zoom, box-zoom, reset');
  const box = await page.$eval('#chartReward', el => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
  const zoomed = async () => (await page.$eval('#chartReward', el => el.textContent)).includes('zoomed');
  await page.mouse.move(box.x + box.w/2, box.y + box.h/2); await page.waitForTimeout(120);
  ok((await page.$eval('#tt', el => getComputedStyle(el).display)) !== 'none', 'hover shows a tooltip');
  await page.mouse.wheel(0, -140); await page.waitForTimeout(150);
  ok(await zoomed(), 'scroll wheel zooms in');
  await page.dblclick('#chartReward'); await page.waitForTimeout(150);
  ok(!(await zoomed()), 'double-click resets zoom');
  await page.mouse.move(box.x + box.w*0.35, box.y + box.h*0.5); await page.mouse.down();
  await page.mouse.move(box.x + box.w*0.65, box.y + box.h*0.5); await page.mouse.up(); await page.waitForTimeout(150);
  ok(await zoomed(), 'drag selects a region to box-zoom');
  await page.click('[data-back="chartReward"]'); await page.waitForTimeout(150);
  ok(!(await zoomed()), 'Back button restores the previous view');

  console.log('\n[M] maximize a chart card');
  await page.click('.card button[title="maximize"]'); await page.waitForTimeout(200);
  ok(await page.isVisible('#chartModal .sheet'), 'chart maximizes into a modal');
  ok((await page.$eval('#chartBig', el => el.querySelectorAll('path').length)) > 0, 'maximized chart renders its series');
  await page.click('#chartClose'); await page.waitForTimeout(150);
  ok(!(await page.isVisible('#chartModal .sheet')), 'maximized chart closes');

  console.log('\n[O] outcomes (bar) chart has a hover tooltip');
  const ob = await page.$eval('#chartOutcomes', el => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
  await page.mouse.move(ob.x + ob.w*0.5, ob.y + ob.h*0.5); await page.waitForTimeout(120);
  ok((await page.$eval('#tt', el => getComputedStyle(el).display)) !== 'none', 'outcomes chart shows a hover tooltip');

  console.log('\n[12] no JS console/page errors');
  ok(consoleErrors.length === 0, `console errors: ${consoleErrors.length ? '\n   - ' + consoleErrors.join('\n   - ') : 'none'}`);

  await browser.close();
  console.log(`\n==== ${failures === 0 ? 'ALL TESTS PASSED' : failures + ' TEST(S) FAILED'} ====`);
  process.exit(failures === 0 ? 0 : 1);
})().catch(e => { console.error('TEST HARNESS ERROR:', e); process.exit(2); });
