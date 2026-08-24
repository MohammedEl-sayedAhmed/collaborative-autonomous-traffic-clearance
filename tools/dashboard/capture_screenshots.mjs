// Capture the dashboard screenshots used in README/docs, driven headlessly.
//   ./run.sh campaign && ./run.sh rl-blocker && ./run.sh dashboard &
//   npm i playwright && npx playwright install chromium
//   DASH_URL=http://127.0.0.1:8770 node tools/dashboard/capture_screenshots.mjs
// Writes PNGs into docs/img/.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import path from 'path';

const URL = process.env.DASH_URL || 'http://127.0.0.1:8770';
const OUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../docs/img');

async function select(page, labels) {
  // read rows as data first (handles go stale when the list re-renders), then
  // click by selector so each click re-queries the DOM.
  const rows = await page.$$eval('.runrow', els => els.map(e => ({
    run: e.getAttribute('data-run'),
    lbl: (e.querySelector('.lbl').firstChild ? e.querySelector('.lbl').firstChild.textContent : '').trim(),
  })));
  await page.click('#selNone'); await page.waitForTimeout(150);
  for (const r of rows) {
    if (labels.includes(r.lbl)) {
      await page.click(`.runrow[data-run="${r.run}"]`); await page.waitForTimeout(150);
    }
  }
  await page.waitForTimeout(900); // allow the per-selection fetch + redraw
}

const shot = (page, name) => page.screenshot({ path: path.join(OUT, name), fullPage: false });

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1680, height: 1050 }, deviceScaleFactor: 2 });
  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.waitForSelector('.runrow', { timeout: 10000 });
  await page.click('#pauseChk'); // stop the auto-poll so captures are deterministic

  const campaign = ['baseline', 'fix-enable-lane-change', 'fix-epsilon-decay', 'fix-randomize-start'];
  const blocker = ['blocker-random', 'blocker-tabular', 'blocker-linear-fa'];

  // hero: the enriched blocker experiment (function approximation solves it)
  await select(page, blocker);            await shot(page, 'dashboard.png');
  // fix-by-fix campaign
  await select(page, campaign);           await shot(page, 'dashboard-campaign.png');
  // live/training view: a single climbing run
  await select(page, ['blocker-linear-fa']); await shot(page, 'dashboard-live.png');
  // single-run detail (used by the guide doc)
  await select(page, ['fix-randomize-start']); await shot(page, 'dashboard-single.png');
  // in-app guide overlay
  await select(page, campaign);
  await page.click('#guideBtn'); await page.waitForTimeout(300);
  await shot(page, 'dashboard-guide.png');
  await page.click('#guideClose');

  await browser.close();
  console.log('wrote screenshots to', OUT);
})().catch(e => { console.error(e); process.exit(1); });
