import { app } from "../../scripts/app.js";

const NODE_ID = "H3StoryDirector";
const PLAN_NODE_ID = "MiniMaxH3ChainPlan";
const PLAN_MARKER = "--- PLAN JSON ---";

function firstText(value) {
    const first = Array.isArray(value) ? value[0] : value;
    return Array.isArray(first) ? first.join("\n") : String(first ?? "");
}

function maskApiKey(node) {
    const widget = node.widgets?.find((item) => item.name === "api_key");
    if (!widget) return;
    widget.options = {...(widget.options ?? {}), password: true};
    for (const element of [widget.inputEl, widget.input, widget.element]) {
        if (element instanceof HTMLInputElement) {
            element.type = "password";
            element.autocomplete = "off";
            element.spellcheck = false;
        }
    }
}

function connectedPlanNodes(node) {
    const plans = [];
    for (const linkId of node.outputs?.[0]?.links ?? []) {
        const link = node.graph?.links?.[linkId];
        const target = node.graph?.getNodeById?.(link?.target_id);
        if (target?.comfyClass === PLAN_NODE_ID || target?.type === PLAN_NODE_ID) {
            plans.push(target);
        }
    }
    return plans;
}

function syncPlan(node, planJson) {
    for (const planNode of connectedPlanNodes(node)) {
        const widget = planNode.widgets?.find((item) => item.name === "plan_json");
        if (!widget) continue;
        widget.value = planJson;
        widget.callback?.(planJson);
        planNode._h3ChainEditorRefresh?.();
        planNode.graph?.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "elgalardi.VisionPromptAssistant.H3StoryDirector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            maskApiKey(this);
            requestAnimationFrame(() => maskApiKey(this));

            const textarea = document.createElement("textarea");
            textarea.readOnly = true;
            textarea.placeholder = "The synopsis and validated H3 plan will appear here";
            Object.assign(textarea.style, {
                background: "var(--comfy-input-bg, #222)",
                border: "1px solid var(--border-color, #555)",
                borderRadius: "6px",
                boxSizing: "border-box",
                color: "var(--input-text, #ddd)",
                fontFamily: "ui-monospace, monospace",
                fontSize: "11px",
                height: "100%",
                lineHeight: "1.4",
                padding: "8px",
                resize: "none",
                width: "100%",
            });
            this.h3StoryDirectorPreview = textarea;
            this.addDOMWidget("story_preview", "h3_story_preview", textarea, {
                serialize: false,
                getMinHeight: () => 180,
            });
            this.setSize([
                Math.max(this.size[0], 390),
                Math.max(this.size[1], 720),
            ]);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const text = firstText(message?.text);
            if (this.h3StoryDirectorPreview) {
                this.h3StoryDirectorPreview.value = text;
                this.h3StoryDirectorPreview.title = text;
            }
            const marker = text.indexOf(PLAN_MARKER);
            if (marker >= 0) {
                const planJson = text.slice(marker + PLAN_MARKER.length).trim();
                syncPlan(this, planJson);
            }
            this.graph?.setDirtyCanvas(true, true);
            return result;
        };
    },
});
