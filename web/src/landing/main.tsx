import { StrictMode } from "react";
import { hydrateRoot } from "react-dom/client";
import "@fontsource-variable/inter/wght.css";
import "@fontsource/jetbrains-mono/400.css";
import "../index.css";
import Landing from "./Landing";

// The HTML is pre-rendered at build time (crawlers read it without JS); this makes the demos interactive.
hydrateRoot(
  document.getElementById("root")!,
  <StrictMode>
    <Landing />
  </StrictMode>,
);
