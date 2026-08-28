// Valida a lógica de paginação da galeria de mamografias (5 colunas x 3 linhas = 15/página)
const samples = [];
for (let i = 0; i < 90; i++) {
  samples.push({
    id: `test_${i}`,
    label: i % 2 === 0 ? 'benign' : 'malignant',
    split: 'test',
    image_url: `/predict/image/samples/test_${i}`,
  });
}
for (let i = 0; i < 375; i++) samples.push({ id: `train_${i}`, label: 'benign', split: 'train', image_url: '/x' });
for (let i = 0; i < 94; i++) samples.push({ id: `val_${i}`, label: 'malignant', split: 'val', image_url: '/x' });

const PER_PAGE = 15;
const COLS = 5;

function totalPages(filtered) { return Math.max(1, Math.ceil(filtered.length / PER_PAGE)); }
function pageItems(filtered, page) {
  const start = (page - 1) * PER_PAGE;
  return filtered.slice(start, start + PER_PAGE);
}

let failures = 0;
function check(name, cond) {
  console.log((cond ? 'OK  ' : 'FALHA') + ' ' + name);
  if (!cond) failures++;
}

// Todas: 559 imagens -> 38 páginas (ceil(559/15))
check('todas: 559 imagens -> 38 páginas', totalPages(samples) === 38);
check('todas: página 1 tem 15 itens', pageItems(samples, 1).length === 15);
check('todas: última página tem 4 itens (559 - 37*15)', pageItems(samples, 38).length === 4);

// Teste: 90 imagens -> 6 páginas
const test = samples.filter(s => s.split === 'test');
check('teste: 90 imagens -> 6 páginas', totalPages(test) === 6);
check('teste: página 1 com 15 itens = 3 linhas x 5 colunas', pageItems(test, 1).length === 15 && COLS === 5);
check('teste: página 6 com 15 itens', pageItems(test, 6).length === 15);

// Treino: 375 -> 25 páginas
const train = samples.filter(s => s.split === 'train');
check('treino: 375 imagens -> 25 páginas', totalPages(train) === 25);

// Val: 94 -> 7 páginas (ceil(94/15)=7), última com 4
const val = samples.filter(s => s.split === 'val');
check('val: 94 imagens -> 7 páginas', totalPages(val) === 7);
check('val: última página tem 4 itens', pageItems(val, 7).length === 4);

// Cada item renderizado tem onclick com id escapado
const item = pageItems(test, 1)[0];
const html = `<div onclick="predictMammoGallerySample('${item.id.replace(/'/g, "\\'")}', this)">`;
check('onclick contém id', html.includes(item.id));

console.log(failures === 0 ? '\nTODAS AS CHECAGENS PASSARAM' : `\n${failures} FALHAS`);
process.exit(failures === 0 ? 0 : 1);
