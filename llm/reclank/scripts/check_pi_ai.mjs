/**
 * Check the server through pi-ai, the client library DeepSeek Harness uses.
 *
 *     node scripts/check_pi_ai.mjs [baseUrl]
 *
 * It streams one tool turn and one plain turn and prints the events that the
 * harness sees: the thinking block, the tool call, and the usage. It needs the
 * harness checkout, because it imports the installed pi-ai.
 */
import { stream } from '/home/cpage/src/deepseek-harness/packages/llm/llm-pi-ai/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js';

const base = process.argv[2] || 'http://jackal.local:8123/v1';
const model = {
  id: 'gemma-4-26B_q4_0-it', name: 'Gemma 4 26B', api: 'openai-completions',
  provider: 'npgemma', baseUrl: base, reasoning: true, input: ['text'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 8192, maxTokens: 4096,
};
const context = {
  messages: [{ role: 'user', content: 'Read the file /etc/hosts. Use the tool.' }],
  tools: [{ name: 'read_file', description: 'Read a file',
            parameters: { type: 'object',
                          properties: { path: { type: 'string', description: 'The file path' } },
                          required: ['path'] } }],
};

async function run(label, ctx, opts) {
  const result = stream(model, ctx, opts);
  const kinds = {};
  let final = null;
  for await (const ev of result) {
    kinds[ev.type] = (kinds[ev.type] || 0) + 1;
    if (ev.type === 'done') final = ev.message;
    if (ev.type === 'error') console.log(label, 'ERROR', ev.error && ev.error.message);
  }
  console.log(label, 'events', JSON.stringify(kinds));
  if (!final) return false;
  console.log(label, 'stopReason', final.stopReason);
  for (const b of final.content) {
    if (b.type === 'text') console.log(label, 'text', JSON.stringify(b.text));
    if (b.type === 'thinking') console.log(label, 'thinking', JSON.stringify(b.thinking.slice(0, 90)));
    if (b.type === 'toolCall') console.log(label, 'toolCall', b.name, JSON.stringify(b.arguments));
  }
  console.log(label, 'usage', JSON.stringify(final.usage));
  return true;
}

const a = await run('tools', context, { maxTokens: 128, temperature: 0, reasoningEffort: 'high', apiKey: 'local' });
const b = await run('plain', { messages: [{ role: 'user', content: 'Say hi in three words.' }] },
                    { maxTokens: 24, temperature: 0, apiKey: 'local' });
console.log(a && b ? 'PASS' : 'CHECK OUTPUT');
