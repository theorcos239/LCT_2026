// Сервис-воркер веб-приложения «Контроль качества DXA».
//
// Нужен для двух вещей: браузер предлагает «Установить приложение», и при
// остановленном сервисе открывается понятная страница, а не ошибка браузера.
// Снимки и результаты не кэшируются: всё, кроме этой страницы и иконки, идёт
// в сеть к сервису как обычно.
const CACHE = 'dxa-qc-shell-v1';
const OFFLINE = '/static/offline.html';

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll([OFFLINE, '/static/icons/icon-192.png'])));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  if (e.request.mode !== 'navigate') return;
  e.respondWith(fetch(e.request).catch(() => caches.match(OFFLINE)));
});
