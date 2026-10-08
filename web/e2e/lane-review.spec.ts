import { expect, test, type Page } from '@playwright/test';

function map(count: number) {
  return {format:'vectormap-ir',version:1,
    lanes:Array.from({length:count},(_,i)=>({id:i+1,kind:'driving',left:100+i*2,right:101+i*2})),
    boundaries:Array.from({length:count*2},(_,i)=>({id:100+i,kind:{type:'virtual'},geometry:[[Math.floor(i/2)*40,i%2?-1.75:1.75,2],[Math.floor(i/2)*40+10,i%2?-1.75:1.75,2]]})),
  };
}
async function open(page: Page, count: number) {
  await page.locator('#vm-file').setInputFiles({name:'map.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(map(count)))});
  await expect(page.locator('#vm-status')).toContainText(`${count} lane`);
}
async function csv(page: Page) {
  const download=page.waitForEvent('download',d=>d.suggestedFilename()==='lane-reviews.csv');
  await page.locator('#vm-review-export').click();
  const chunks: Buffer[]=[];
  for await (const chunk of await (await download).createReadStream()) chunks.push(chunk);
  return Buffer.concat(chunks).toString('utf8');
}

test('review list pages, selects lanes, filters saved decisions and exports every lane', async ({page},info) => {
  test.setTimeout(120_000);
  const errors:string[]=[]; page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#vm-review-export')).toBeDisabled();
  await expect(page.locator('#vm-review-next')).toBeDisabled();
  await open(page,30);
  await expect(page.locator('#vm-review-rows tr')).toHaveCount(25);
  await expect(page.locator('#vm-review-page')).toContainText('Showing 1–25 of 30');
  await page.locator('#vm-review-page-next').click();
  await expect(page.locator('#vm-review-rows tr')).toHaveCount(5);
  await page.getByRole('button',{name:'View lane 26',exact:true}).click();
  await expect(page.locator('#vm-lane-title')).toHaveText('Lane 26');
  await expect(page.getByRole('button',{name:'View lane 26',exact:true})).toHaveAttribute('aria-current','true');
  await page.locator('#vm-review-state').selectOption('needs-fix');
  await page.locator('#vm-review-notes').fill('幅, "縁石"\n再計測');
  await page.locator('#vm-review-save').click();
  await page.locator('#vm-review-filter').selectOption('needs-fix');
  await expect(page.locator('#vm-review-rows tr')).toHaveCount(1);
  await expect(page.locator('#vm-review-rows')).toContainText('Needs fixes');
  await expect(page.locator('#vm-review-rows')).toContainText('再計測');
  await expect(page.locator('#vm-review-page-next')).toBeDisabled();
  await expect(page.locator('#vm-review-page-prev')).toBeDisabled();
  await page.locator('#vm-review-filter').selectOption('all');
  await page.locator('#vm-review-next').click();
  await expect(page.locator('#vm-lane-title')).toHaveText('Lane 27');
  await expect(page.locator('#vm-review-page')).toContainText('Showing 26–30 of 30');
  await expect(page.getByRole('button',{name:'View lane 27',exact:true})).toHaveAttribute('aria-current','true');
  await page.locator('#vm-review-notes').fill('Unsaved scratch note');
  const output=await csv(page);
  expect(output.startsWith('\uFEFFlane_id,status,notes,updated_at,previous_status,stale_reason\r\n')).toBe(true);
  expect(output.split('\r\n').filter(Boolean)).toHaveLength(31);
  expect(output).toContain('26,"needs-fix","幅, ""縁石""\n再計測"');
  expect(output).toContain('27,"unreviewed","","","",""\r\n');
  expect(output).not.toContain('Unsaved scratch note');
  // Export all lanes even while the list shows only one saved decision.
  await page.locator('#vm-review-filter').selectOption('needs-fix');
  expect(await csv(page)).toBe(output);
  await page.getByRole('button',{name:'View lane 26',exact:true}).click();
  await page.locator('#vm-review-panel').scrollIntoViewIfNeeded();
  await page.screenshot({path:info.outputPath('lane-review-list.png')});
  expect(errors).toEqual([]);
});

test('list and CSV reflect invalidation, deletion and opening a different map', async ({page}) => {
  const errors:string[]=[]; page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/'); await open(page,3);
  await page.getByRole('button',{name:'View lane 1',exact:true}).click();
  await page.locator('#vm-review-state').selectOption('reviewed');
  const notes='<img src=x onerror="alert(1)">';
  await page.locator('#vm-review-notes').fill(notes);
  await page.locator('#vm-review-save').click();
  await page.locator('#vm-review-filter').selectOption('reviewed');
  await expect(page.locator('#vm-review-rows tr')).toHaveCount(1);
  await expect(page.locator('#vm-review-rows')).toContainText(notes);
  await expect(page.locator('#vm-review-rows img')).toHaveCount(0);
  await page.locator('#vm-lane-speed').fill('30');
  await page.locator('#vm-lane-apply').click();
  await expect(page.locator('#vm-review-state')).toHaveValue('unreviewed');
  await expect(page.locator('#vm-review-rows tr')).toHaveCount(0);
  await expect(page.locator('#vm-review-empty')).toContainText('No lanes match');
  await page.locator('#vm-review-filter').selectOption('unreviewed');
  const row=page.locator('#vm-review-rows tr',{has:page.getByRole('button',{name:'View lane 1',exact:true})});
  await expect(row).toContainText('Review again');
  const output=await csv(page);
  expect(output).toContain('"reviewed","Lane geometry or associated rules changed; review again."');
  await page.locator('#vm-lane-delete').click();
  await expect(page.locator('#status')).toContainText('Lane 1 removed');
  await expect(page.locator('#vm-review-rows tr')).toHaveCount(2);
  expect(await csv(page)).not.toMatch(/\r\n1,/);
  await open(page,1);
  await expect(page.locator('#vm-review-rows')).not.toContainText('Review again');
  await expect(page.locator('#vm-review-rows')).not.toContainText(notes);
  expect(await csv(page)).toContain('1,"unreviewed","","","",""\r\n');
  await page.locator('#vm-review-filter').selectOption('low-support');
  await expect(page.locator('#vm-review-rows tr')).toHaveCount(0);
  await expect(page.locator('#vm-review-empty')).toContainText('Check source coverage');
  expect(errors).toEqual([]);
});
