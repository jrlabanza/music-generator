/* Music Gen Studio service worker: makes the page installable and keeps the shell available
   while the network is slow. API calls and audio are never cached (always network). */
const CACHE = "mgs-shell-v1";
const SHELL = ["/", "/static/style.css", "/static/app.js", "/static/vendor/abcjs-basic-min.js", "/manifest.webmanifest",
               "/static/icons/icon-192.png", "/static/icons/icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL).catch(() => null)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== location.origin) return;
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/outputs/") || url.pathname.startsWith("/s/") || url.pathname === "/login") return;
  // network first, cache as fallback, so updates show up immediately when online
  event.respondWith(fetch(event.request).then((response) => {
    if (response.ok && (SHELL.includes(url.pathname) || url.pathname.startsWith("/static/"))) {
      const copy = response.clone();
      caches.open(CACHE).then((cache) => cache.put(event.request, copy));
    }
    return response;
  }).catch(() => caches.match(event.request).then((hit) => hit || Response.error())));
});
