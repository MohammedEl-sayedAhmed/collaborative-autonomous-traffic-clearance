// Optional Playwright regression test for the dashboard's run-selection logic.
// It drives a real browser and asserts that, after many click sequences, the
// checkbox state, the row highlight, and the comparison-table membership always
// agree (the bug this guards against was competing "focus" vs. checkbox state).
//
// Run it against a live dashboard with demo data:
//   ./run.sh dashboard-demo          # seed two runs
//   ./run.sh dashboard &             # serve at :8770
//   npm i playwright && npx playwright install chromium
//   DASH_URL=http://127.0.0.1:8770 node tools/dashboard/test_dashboard.mjs
//
// Exit code 0 = all passed.
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

async function invariant(page, expected, tag) {
  const r = await rows(page);
  const checked = r.filter(x => x.checked).map(x => x.label).sort();
  const selClass = r.filter(x => x.sel).map(x => x.label).sort();
  const table = (await tableLabels(page)).sort();
  const exp = [...expected].sort();
  ok(eq(checked, exp), `${tag}: checkboxes = [${checked}] (expected [${exp}])`);
  ok(eq(selClass, exp), `${tag}: .sel highlight = [${selClass}]`);
  ok(eq(table, exp), `${tag}: comparison table = [${table}]`);
}

const consoleErrors = [];
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on('pageerror', e => consoleErrors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') consoleErrors.push('console.error: ' + m.text()); });

  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.waitForSelector('.runrow', { timeout: 10000 });
  await page.click('#pauseChk'); // deterministic: stop the 2s auto-poll

  const r0 = await rows(page);
  console.log(`Found ${r0.length} runs: ${r0.map(x => x.label).join(', ')}`);
  ok(r0.length >= 2, 'at least 2 runs present');
  const [A, B] = r0;
  const click = async (run) => { await page.click(`.runrow[data-run="${run}"]`); await page.waitForTimeout(120); };

  await page.click('#selNone'); await page.waitForTimeout(120); await invariant(page, [], 'none');
  await click(A.run); await invariant(page, [A.label], 'A on');
  await click(A.run); await invariant(page, [], 'A off');
  await click(A.run); await click(B.run); await invariant(page, [A.label, B.label], 'A+B');
  await click(A.run); await invariant(page, [B.label], 'only B');
  for (let i = 0; i < 5; i++) await click(A.run);  await invariant(page, [A.label, B.label], 'rapid5 A -> on');
  for (let i = 0; i < 4; i++) await click(B.run);  await invariant(page, [A.label, B.label], 'rapid4 B -> on');
  await page.click('#selNone'); await page.waitForTimeout(120);
  await click(A.run); await click(B.run); await click(A.run); await click(B.run); await click(B.run);
  await invariant(page, [B.label], 'interleaved A,B,A,B,B');
  await page.click('#selAll'); await page.waitForTimeout(120); await invariant(page, r0.map(x => x.label), 'all');
  await page.click('#selNone'); await page.waitForTimeout(120); await invariant(page, [], 'none again');

  await click(A.run); await click(B.run); await page.waitForTimeout(600);
  ok((await legendCount(page)) === 2, 'reward legend shows 2 series');
  ok(/Runs compared|Total episodes/.test(await page.$eval('#tiles', el => el.textContent)), 'tiles show multi-run summary when 2 selected (no lag)');

  // in-dashboard guide panel
  await page.click('#guideBtn'); await page.waitForTimeout(150);
  ok(await page.isVisible('#guide .sheet'), 'guide opens');
  const gtext = await page.$eval('#guide .sheet', el => el.textContent);
  ok(gtext.includes('Q-table') && gtext.includes('Episode') && gtext.includes('Epsilon'), 'guide explains RL/Q-table/episode/epsilon');
  await page.click('#guideClose'); await page.waitForTimeout(150);
  ok(!(await page.isVisible('#guide .sheet')), 'guide closes');

  // responsive layout: 4 charts in one row at 2K, no horizontal page scroll
  await page.setViewportSize({ width: 2560, height: 1440 }); await page.waitForTimeout(250);
  const tops = await page.$$eval('.grid2 .card', els => els.map(e => Math.round(e.getBoundingClientRect().top)));
  ok(new Set(tops).size === 1, `4 charts share one row at 2560px (rows=${new Set(tops).size})`);
  const hScroll = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
  ok(!hScroll, 'no horizontal page scroll at 2560px');

  ok(consoleErrors.length === 0, `no console errors${consoleErrors.length ? ':\n   - ' + consoleErrors.join('\n   - ') : ''}`);

  await browser.close();
  console.log(`\n==== ${failures === 0 ? 'ALL TESTS PASSED' : failures + ' TEST(S) FAILED'} ====`);
  process.exit(failures === 0 ? 0 : 1);
})().catch(e => { console.error('TEST HARNESS ERROR:', e); process.exit(2); });
