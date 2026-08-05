import { app } from "../../scripts/app.js";

const NODE_ID = "PreviewVisionPrompt";
const MIN_HEIGHT = 120;

function firstText(value) {
    const first = Array.isArray(value) ? value[0] : value;
    return Array.isArray(first) ? first.join("\n") : String(first ?? "");
}

app.registerExtension({
    name: "elgalardi.LocalVisionPrompt.Preview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const textarea = document.createElement("textarea");
            textarea.readOnly = true;
            textarea.placeholder = "The generated vision prompt will appear here";
            Object.assign(textarea.style, {
                background: "var(--comfy-input-bg, #222)",
                border: "1px solid var(--border-color, #555)",
                borderRadius: "6px",
                boxSizing: "border-box",
                color: "var(--input-text, #ddd)",
                fontFamily: "system-ui, sans-serif",
                fontSize: "12px",
                height: "100%",
                lineHeight: "1.4",
                padding: "8px",
                resize: "none",
                width: "100%",
            });

            this.previewVisionPromptElement = textarea;
            this.addDOMWidget("preview_text", "preview_vision_prompt", textarea, {
                serialize: false,
                getMinHeight: () => MIN_HEIGHT,
            });

            this.setSize([
                Math.max(this.size[0], 320),
                Math.max(this.size[1], 220),
            ]);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            if (this.previewVisionPromptElement) {
                const text = firstText(message?.text);
                this.previewVisionPromptElement.value = text;
                this.previewVisionPromptElement.title = text;
                this.graph?.setDirtyCanvas(true, true);
            }
            return result;
        };
    },
});
