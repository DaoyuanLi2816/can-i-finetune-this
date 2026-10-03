const form = document.querySelector('#config');
const model = document.querySelector('#model');
const results = document.querySelector('#results');
const error = document.querySelector('#error');
function element(tag, text, className) { const el = document.createElement(tag); el.textContent = text; if (className) el.className = className; return el; }
async function estimate(event) {
  if (event) event.preventDefault();
  error.textContent = '';
  const input = Object.fromEntries(new FormData(form));
  for (const key of ['gpu_vram_gb', 'seq_len', 'micro_batch_size', 'lora_rank']) input[key] = Number(input[key]);
  input.available_vram_gb = input.available_vram_gb ? Number(input.available_vram_gb) : null;
  input.gradient_checkpointing = form.elements.gradient_checkpointing.checked;
  try {
    const response = await fetch('/api/estimate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(input)});
    const data = await response.json(); if (!response.ok) throw new Error(data.error);
    results.replaceChildren(element('h2', 'Your preflight result'));
    results.append(element('div', `${data.feasible.toUpperCase()} · ${data.memory.total_estimated_gb.toFixed(2)} GiB planning budget`, `verdict ${data.feasible}`));
    results.append(element('p', `Budget ratio ${(data.feasibility_ratio * 100).toFixed(0)}% · confidence ${data.confidence}`));
    const table = element('table', ''); const tbody = element('tbody', '');
    for (const [key, value] of Object.entries(data.memory)) { if (key === 'trainable_params_mb') continue; const row = element('tr', ''); row.append(element('td', key.replaceAll('_gb', '').replaceAll('_', ' ')), element('td', value.toFixed(3) + ' GiB')); tbody.append(row); } table.append(tbody); results.append(table);
    results.append(element('h3', 'Run the same estimate'), element('pre', data.command), element('h3', 'Generate a recipe'), element('pre', data.recipe_command));
    results.append(element('h3', 'Assumptions and evidence'));
    for (const text of [...data.assumptions, ...data.warnings, data.confidence_basis, data.measurement_target]) results.append(element('p', text, 'note'));
  } catch (exc) { error.textContent = exc.message; results.replaceChildren(element('h2', 'Your preflight result'), element('p', 'Correct the configuration to calculate a new result.')); }
}
form.addEventListener('submit', estimate);
fetch('/api/models').then(r => r.json()).then(models => { for (const id of models) model.append(element('option', id)); model.value = 'Qwen/Qwen2.5-1.5B-Instruct'; estimate(); }).catch(exc => { error.textContent = exc.message; });
