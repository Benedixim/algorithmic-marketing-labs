import { pipeline, env } from '@huggingface/transformers';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
env.cacheDir=process.env.MARKETING_MODEL_CACHE || path.join(ROOT,'model-cache');
const generator=await pipeline('text-generation','onnx-community/Qwen2.5-1.5B-Instruct',{dtype:'q4',device:'cpu'});
const segments=JSON.parse(await fs.readFile(path.join(ROOT,'results','lab09_inputs.json'),'utf8'));
const results=[];
for(const s of segments){
 const opening={'Лояльные':'For your next order,','Новые':'Welcome to our gift shop,','Под риском':'We would love to see you again,','Спящие':'If you feel like browsing again,','Остальные':'On your next visit,'}[s.segment];
 const prompt=`Write one short English ad. Start exactly with: "${opening}". Then invite the reader to browse this product: "${s.product}". Use exactly the same product name. The audience is ${s.audience}. Only invite browsing. No descriptions, no features, no benefits, no price, no discounts, no delivery, no superlatives. Keep it under 25 words. Return only the final ad.`;
 const start=Date.now();
 const out=await generator([{role:'system',content:'You follow the requested opening and product name exactly. Write a simple invitation, without adjectives or invented details.'},{role:'user',content:prompt}],{max_new_tokens:65,do_sample:false,repetition_penalty:1.15});
 const text=out[0].generated_text.at(-1).content;
 const forbidden=/[$£€%0-9]|discount|free shipping|perfect|ultimate|stunning|unmatched|premium|exclusive|latest collection|jewelry/i;
 const normalized=text.toUpperCase().replace(/[^A-Z0-9]+/g,' ').trim();
 const product=s.product.toUpperCase().replace(/[^A-Z0-9]+/g,' ').trim();
 const checks={product:normalized.includes(product),noUnsupportedClaim:!forbidden.test(text),short:text.trim().split(/\s+/).length<=30};
 const accepted=Object.values(checks).every(Boolean);
 results.push({...s,prompt,text,checks,accepted,seconds:(Date.now()-start)/1000});
 console.log(JSON.stringify({segment:s.segment,text}));
}
await fs.writeFile(path.join(ROOT,'results','lab09_generation.json'),JSON.stringify({model:'onnx-community/Qwen2.5-1.5B-Instruct',dtype:'q4',runtime:'Transformers.js 4.3.2 / CPU',results},null,2));
await generator.dispose();
