function renderKatex() {
  if (typeof renderMathInElement === "undefined") {
    return;
  }

  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false }
    ],
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    renderKatex();
  });
} else {
  document.addEventListener("DOMContentLoaded", () => {
    renderKatex();
  });
}
