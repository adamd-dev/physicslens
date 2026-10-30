import { StrictMode } from "react";
import { renderToString } from "react-dom/server";
import { FAQ } from "../home/Faq";
import Landing from "./Landing";

export function render(): string {
  return renderToString(
    <StrictMode>
      <Landing />
    </StrictMode>,
  );
}

/** Structured data built from the visible FAQ, so the two always say the same thing. */
export function jsonLd(siteUrl: string): string {
  const data = {
    "@context": "https://schema.org",
    "@graph": [
      { "@type": "WebSite", "@id": `${siteUrl}/#website`, url: `${siteUrl}/`, name: "PhysicsLens", inLanguage: "en" },
      {
        "@type": "SoftwareApplication",
        "@id": `${siteUrl}/#app`,
        name: "PhysicsLens",
        url: `${siteUrl}/`,
        image: `${siteUrl}/og-image.png`,
        description:
          "Measures gravitational acceleration, pendulum length and landing point from a phone video of a physics experiment. Ball tracking uses motion and color trackers or Meta SAM 3.1, SciPy fits the physics model, and a Meta Muse Spark agent checks the fit.",
        applicationCategory: "EducationalApplication",
        applicationSubCategory: "Physics",
        operatingSystem: "macOS, Linux, Windows (Python 3, runs locally)",
        isAccessibleForFree: true,
        offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
        codeRepository: "https://github.com/adamd-dev/physicslens",
        author: { "@type": "Person", name: "Adam D.", url: "https://github.com/adamd-dev" },
        featureList: [
          "Measure g from a ball drop video",
          "Measure pendulum length from a swing video",
          "Predict the landing point of a thrown ball",
          "A Meta Muse Spark agent checks every fit",
        ],
        inLanguage: "en",
      },
      {
        "@type": "FAQPage",
        mainEntity: FAQ.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })),
      },
    ],
  };
  // "<" escaped so no answer text can close the script element.
  return `<script type="application/ld+json">${JSON.stringify(data).replace(/</g, "\\u003c")}</script>`;
}
