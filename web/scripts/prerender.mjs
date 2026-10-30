// Fills dist-landing/index.html with the server-rendered page and its JSON-LD, and the site URL everywhere.
import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const site = (process.env.SITE_URL || "https://physicslens.vercel.app").replace(/\/$/, "");
const { render, jsonLd } = await import(pathToFileURL(`${root}dist-landing-ssr/server.js`).href);

const file = `${root}dist-landing/landing.html`;
const html = readFileSync(file, "utf8")
  .replace("<!--app-->", render())
  .replace("<!--jsonld-->", jsonLd(site))
  .replaceAll("__SITE_URL__", site);
writeFileSync(`${root}dist-landing/index.html`, html);
rmSync(file);
for (const name of ["robots.txt", "sitemap.xml"]) {
  const path = `${root}dist-landing/${name}`;
  writeFileSync(path, readFileSync(path, "utf8").replaceAll("__SITE_URL__", site));
}
rmSync(`${root}dist-landing-ssr`, { recursive: true, force: true });
console.log(`landing pre-rendered for ${site}`);
