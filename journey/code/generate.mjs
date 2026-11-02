// Receives aggregate segment portraits only. No customer identities are sent.
import {pipeline,env} from '@huggingface/transformers';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
env.cacheDir=process.env.MARKETING_MODEL_CACHE||path.join(root,'model-cache');
const model='onnx-community/Qwen2.5-1.5B-Instruct';
const generator=await pipeline('text-generation',model,{dtype:'q4',device:'cpu'});
const inputs=JSON.parse(await fs.readFile(path.join(root,'results/lab09_inputs.json'),'utf8'));
const openings={
 'Лучшие':'Thanks for shopping with us.',
 'Под риском':'Ready to plan your next grocery trip?',
 'Неактивные':'Take a look at the grocery selection.',
 'Недавние редкие':'Plan your next grocery basket.',
 'Остальные':'Time to restock your usual groceries?'
};
const results=[];
for(const s of inputs){
 const category=s.category.toLowerCase();
 const opening=openings[s.segment];
 const prompt=`Write one short English grocery shopping invitation. Audience: ${s.audience}. Goal: ${s.goal}. Start exactly with "${opening}". Then invite the reader to browse ${category}. Only invite browsing. Do not describe product properties. No discounts, prices, gifts, delivery promises, superlatives or invented offers. Under 30 words. No emojis or hashtags. Return the message only.`;
 const start=Date.now();
 const out=await generator([{role:'system',content:'Write a plain invitation using only the supplied facts. Follow the requested opening exactly.'},{role:'user',content:prompt}],{do_sample:false,max_new_tokens:90,repetition_penalty:1.12});
 const raw=out[0].generated_text.at(-1).content.trim();
 const checks={short:raw.split(/\s+/).length<=30,category:raw.toLowerCase().includes(category),opening:raw.startsWith(opening),noDecorations:!/[#\p{Extended_Pictographic}]/u.test(raw),noUnsupportedClaim:!/(discount|free|gift|guarantee|best|exclusive|delivery|shipping|premium|%|\d)/i.test(raw)};
 results.push({...s,prompt,text:raw,checks,accepted:Object.values(checks).every(Boolean),seconds:(Date.now()-start)/1000});
 console.log(JSON.stringify({segment:s.segment,text:raw,checks}));
}
await fs.writeFile(path.join(root,'results/lab09_generation.json'),JSON.stringify({model,runtime:'Transformers.js / CPU / q4',results},null,2));
await generator.dispose();
