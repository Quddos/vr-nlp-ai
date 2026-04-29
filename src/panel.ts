import {
  createSystem,
  PanelUI,
  PanelDocument,
  eq,
  VisibilityState,
  UIKitDocument,
  UIKit,
} from "@iwsdk/core";

export class PanelSystem extends createSystem({
  welcomePanel: {
    required: [PanelUI, PanelDocument],
    where: [eq(PanelUI, "config", "./ui/welcome.json")],
  },
}) {
  init() {
    this.queries.welcomePanel.subscribe("qualify", (entity) => {
      const document = PanelDocument.data.document[
        entity.index
      ] as UIKitDocument;
      if (!document) {
        return;
      }

      const xrButton = document.getElementById("xr-button") as UIKit.Text;
      xrButton.addEventListener("click", () => {
        if (this.world.visibilityState.value === VisibilityState.NonImmersive) {
          this.world.launchXR();
        } else {
          this.world.exitXR();
        }
      });
      this.world.visibilityState.subscribe((visibilityState) => {
        if (visibilityState === VisibilityState.NonImmersive) {
          xrButton.setProperties({ text: "Enter XR" });
        } else {
          xrButton.setProperties({ text: "Exit to Browser" });
        }
      });

      // Translation functionality
      const translateButton = document.getElementById("translate-button") as UIKit.Text;
      const inputText = document.getElementById("input-text") as UIKit.Text;
      const outputText = document.getElementById("output-text") as UIKit.Text;

      translateButton.addEventListener("click", async () => {
        const text = (inputText as any).value;
        if (!text.trim()) {
          outputText.setProperties({ text: "Please enter some text to translate." });
          return;
        }

        outputText.setProperties({ text: "Translating..." });

        try {
          const response = await fetch('http://localhost:5000/translate', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sentence: text }),
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          outputText.setProperties({ text: data.translation || "Translation failed." });
        } catch (error) {
          console.error('Translation error:', error);
          outputText.setProperties({ text: "Error: Could not connect to translation service. Make sure the Python server is running." });
        }
      });
    });
  }
}
