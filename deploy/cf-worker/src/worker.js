// Cloudflare Worker that serves the rvllm docs at solidsf.com/rvllm/*.
//
// Incoming URL                                            -> File served
// ───────────────────────────────────────────────────────     ────────────────────
// solidsf.com/rvllm/                                      -> docs/index.html
// solidsf.com/rvllm/docs/bench.html                       -> docs/bench.html
// solidsf.com/rvllm/docs/paper/rvllm.pdf                  -> docs/paper/rvllm.pdf
//
// The ./site directory (populated by build.sh before deploy) contains a
// docs/ subdirectory that mirrors the repo's docs/ folder, so the URL
// layout matches the repo layout one-to-one after stripping /rvllm.

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    let p = url.pathname;

    // Strip the /rvllm prefix (the Worker is routed at solidsf.com/rvllm/*).
    if (p.startsWith('/rvllm/')) p = p.slice('/rvllm'.length);
    else if (p === '/rvllm') p = '/';

    // Root: serve docs/index.html directly (no redirect, stays on /rvllm/)
    if (p === '/' || p === '') {
      p = '/docs/index.html';
    }

    // Forward to the static-assets binding.
    const assetUrl = new URL(url);
    assetUrl.pathname = p;
    const res = await env.ASSETS.fetch(new Request(assetUrl, request));

    const ct = res.headers.get('content-type') || '';
    const headers = new Headers(res.headers);
    if (ct.includes('text/html')) {
      headers.set('Cache-Control', 'no-store');
      headers.set('CDN-Cache-Control', 'no-store');
    } else {
      headers.set('Cache-Control', 'public, max-age=3600');
    }
    headers.set('X-Served-By', 'rvllm-docs-worker');
    return new Response(res.body, { status: res.status, headers });
  }
};
