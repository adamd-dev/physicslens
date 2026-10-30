import { StrictMode } from "react";
import { hydrateRoot } from "react-dom/client";
import "./fonts.css";
import "@fontsource/jetbrains-mono/latin-400.css";
import "../index.css";
import Landing from "./Landing";

// The HTML is pre-rendered at build time (crawlers read it without JS); this makes the demos interactive.
hydrateRoot(
  document.getElementById("root")!,
  <StrictMode>
    <Landing />
  </StrictMode>,
);
