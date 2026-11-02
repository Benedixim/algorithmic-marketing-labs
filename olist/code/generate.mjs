// Генеративная модель получает только агрегированный портрет сегмента.
import {pipeline,env} from '@huggingface/transformers';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
env.cacheDir=process.env.MARKETING_MODEL_CACHE||path.join(root,'model-cache');
const model='onnx-community/Qwen2.5-1.5B-Instruct';
const generator=await pipeline('text-generation',model,{dtype:'q4',device:'cpu'});
const inputs=JSON.parse(await fs.readFile(path.join(root,'results/lab09_inputs.json'),'utf8'));
const results=[];
for(const s of inputs){
 const opening={'Повторные':'Thank you for coming back.','Новые':'Thank you for your first order.','Под риском':'It has been a while.','Неактивные':'If you feel like visiting again,','Разовые':'Looking for your next purchase?'}[s.segment];
 const category=s.category==='bed_bath_table'?'home textiles':'beauty and health products';
 const audience={'Повторные':'repeat buyers with a recent order','Новые':'buyers with one recent order','Под риском':'previously valuable buyers who have not ordered for a long time','Неактивные':'buyers with one old order','Разовые':'buyers who have ordered once'}[s.segment];
 const prompt=`Write one short English invitation for Olist. Start exactly with "${opening}". Then invite the reader to browse ${category}. The audience is ${audience}. Only invite browsing. No descriptions, no adjectives, no benefits, no discounts, no delivery promises, no prices. Under 25 words. No emojis or hashtags. Return only the message.`;
 const start=Date.now();
 const out=await generator([{role:'system',content:'Follow the requested opening exactly. Write a plain English invitation. Do not add facts or explanations.'},{role:'user',content:prompt}],{do_sample:false,max_new_tokens:65,repetition_penalty:1.12});
 const raw=out[0].generated_text.at(-1).content.trim();
 const checks={short:raw.split(/\s+/).length<=25,category:raw.toLowerCase().includes(category),opening:raw.startsWith(opening),noDecorations:!/[#\p{Extended_Pictographic}]/u.test(raw),noUnsupportedClaim:!/(discount|free|gift|guarantee|best|exclusive|delivery|shipping|premium|%|\d)/i.test(raw)};
 results.push({...s,prompt,text:raw,checks,accepted:Object.values(checks).every(Boolean),seconds:(Date.now()-start)/1000});
 console.log(JSON.stringify({segment:s.segment,text:raw,checks}));
}
await fs.writeFile(path.join(root,'results/lab09_generation.json'),JSON.stringify({model,runtime:'Transformers.js 4.3.2 / CPU / q4',results},null,2));
await generator.dispose();
