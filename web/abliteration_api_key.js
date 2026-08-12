import { app } from "../../scripts/app.js";

const NODE_ID = "AbliterationVisionPrompt";

function maskApiKeyWidget(node) {
    const widget = node.widgets?.find((item) => item.name === "api_key");
    if (!widget) return;

    widget.options = { ...(widget.options ?? {}), password: true };
    for (const element of [widget.inputEl, widget.input, widget.element]) {
        if (element instanceof HTMLInputElement) {
            element.type = "password";
            element.autocomplete = "off";
            element.spellcheck = false;
        }
    }
}

app.registerExtension({
    name: "elgalardi.VisionPromptAssistant.AbliterationApiKey",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            maskApiKeyWidget(this);
            requestAnimationFrame(() => maskApiKeyWidget(this));
            return result;
        };
    },
});
