import { expect, test, type Page } from '@playwright/test';

async function pixels(page: Page): Promise<string> {
  return (await page.locator('#viewport > canvas').screenshot()).toString('base64');
}
async function compare(page: Page, before: string, after: string) {
  return page.evaluate(async ({before, after}) => {
    async function rgba(png: string) {
      const image = new Image(); image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext('2d')!; context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, canvas.width, canvas.height).data;
    }
    const [a, b] = await Promise.all([rgba(before), rgba(after)]);
    let points=0, visible=0, shaded=0, background=0, unchangedBackground=0;
    for(let i=0;i<a.length;i+=4) {
      if(a[i]>210 && a[i+1]>210 && a[i+2]>210) {
        points++; if(b[i]>190 && b[i+1]>190 && b[i+2]>190) visible++;
        if(a[i]-b[i]>35 && a[i+1]-b[i+1]>35 && a[i+2]-b[i+2]>35) shaded++;
      }
      if(a[i]<25 && a[i+1]<25 && a[i+2]<25) {
        // The existing 8-bit linear render target quantizes dark sRGB colors.
        background++; if(Math.abs(a[i]-b[i])<=5 && Math.abs(a[i+1]-b[i+1])<=5 && Math.abs(a[i+2]-b[i+2])<=5) unchangedBackground++;
      }
    }
    return {points,visible,shaded,background,unchangedBackground};
  }, {before, after});
}
async function open(page: Page, points: number[][], size: number) {
  await page.goto('/');
  const text=`ply\nformat ascii 1.0\nelement vertex ${points.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n${points.map(p=>p.join(' ')).join('\n')}\n`;
  await page.locator('#file-input').setInputFiles({name:'edl.ply',mimeType:'application/octet-stream',buffer:Buffer.from(text)});
  await expect(page.locator('#status')).toContainText('Loaded edl.ply');
  await page.locator('[data-view=top]').click();
  await page.locator('#point-size').fill(String(size));
  await page.locator('#edl').uncheck();
}

test('EDL keeps isolated source points visible against empty background', async ({page}, info)=>{
  const points: number[][]=[];
  for(let x=0;x<5;x++) for(let y=0;y<5;y++) points.push([x*10,y*10,2]);
  await open(page,points,2);
  const before=await pixels(page);
  await page.locator('#edl').check();
  await expect.poll(async()=>{
    const stats=await compare(page,before,await pixels(page));
    expect(stats.points).toBeGreaterThan(80);
    return stats.visible/stats.points;
  }).toBeGreaterThan(0.9);
  const after=await pixels(page), stats=await compare(page,before,after);
  expect(stats.unchangedBackground/stats.background).toBeGreaterThan(0.99);
  await info.attach('sparse-visibility', {body:Buffer.from(JSON.stringify(stats)),contentType:'application/json'});
  await page.locator('#viewport > canvas').screenshot({path:info.outputPath('sparse-edl.png')});
});

test('EDL still shades measured depth steps within dense source points', async ({page}, info)=>{
  const points: number[][]=[];
  for(let x=-60;x<=60;x++) for(let y=-60;y<=60;y++) points.push([x/60,y/60,x<0?0:0.4]);
  await open(page,points,8);
  const before=await pixels(page);
  await page.locator('#edl').check();
  await expect.poll(async()=>{
    const stats=await compare(page,before,await pixels(page));
    expect(stats.points).toBeGreaterThan(1000);
    return stats.shaded;
  }).toBeGreaterThan(50);
  const stats=await compare(page,before,await pixels(page));
  await info.attach('depth-step-shading', {body:Buffer.from(JSON.stringify(stats)),contentType:'application/json'});
  await page.locator('#viewport > canvas').screenshot({path:info.outputPath('depth-step-edl.png')});
});
