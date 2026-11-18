import { app } from "/scripts/app.js";

// Removing the first two widgets must not shift old saved prompts into modes.
// Credentials are discarded, never moved to another node or logged.
export function migrateCompactProviderNode(node) {
    if (node?.type !== "H3CompactMultimodalEditDirector") return;
    const values = node.widgets_values;
    const modes = new Set(["compact", "deep_edit", "continuous", "edit", "Enhance", "Edit Continuo", "Continuous Edit"]);
    if (Array.isArray(values) && modes.has(values[3]) && !modes.has(values[1])) {
        node.widgets_values = values.slice(2);
    }
    const named = node.widgets_values_named;
    if (named && typeof named === "object") {
        delete named.api_key;
        delete named.model;
    }
    const rename = value => ({deep_edit: "edit", continuous: "Enhance", "Edit Continuo": "Continuous Edit",
        "Continuous Elaborate": "Elaborate Plus", "Elaborate Continuo": "Elaborate Plus",
        continuous_elaborate: "Elaborate Plus"})[value] ?? value;
    if (Array.isArray(node.widgets_values)) {
        node.widgets_values[1] = rename(node.widgets_values[1]);
    }
    if (named?.edit_mode) named.edit_mode = rename(named.edit_mode);
}

// Promote proven nodes without changing saved connections or Fix widget order.
export function migratePromotedNode(node) {
    const aliases = {
        H3CompactMultimodalEditDirectorFix: "H3CompactMultimodalEditDirector",
        H3CompactDirectionControlsFix: "H3CompactDirectionControls",
    };
    const promoted = aliases[node?.type];
    if (promoted) {
        node.type = promoted;
        if (node.title?.endsWith(" — Fix")) node.title = node.title.slice(0, -6);
        return;
    }
    if (node?.type === "H3CompactDirectionControls" && Array.isArray(node.widgets_values)
            && node.widgets_values.length === 10) {
        const [genre, , motion, , look, , dialogue, wardrobe, environment, body] = node.widgets_values;
        node.widgets_values = [genre, "none", motion, look, dialogue, "Auto", "none",
            wardrobe, "Auto", environment, body, "none", "Swap"];
    }
    if (node?.type === "H3CompactMultimodalEditDirector") {
        const modes = { Enhance: "Normal", Elaborate: "Normal", "Elaborate Plus": "Normal",
            "Continuous Elaborate": "Normal", "Elaborate Continuo": "Normal",
            continuous_elaborate: "Normal", "Continuous Edit": "Edit", "Edit Continuo": "Edit",
            compact: "Compact", edit: "Edit", deep_edit: "Edit", continuous: "Normal" };
        if (Array.isArray(node.widgets_values)) {
            node.widgets_values[1] = modes[node.widgets_values[1]] ?? node.widgets_values[1];
            // Retired Driven Strength and Creative Profile were the final two widgets.
            if (["Standard", "Creative Plus"].includes(node.widgets_values.at(-1))
                    && Number.isInteger(node.widgets_values.at(-2))) node.widgets_values.splice(-2, 2);
        }
        const named = node.widgets_values_named;
        if (named?.edit_mode) named.edit_mode = modes[named.edit_mode] ?? named.edit_mode;
    }
}

app.registerExtension({
    name: "h3.compact.externalProviderMigration",
    beforeConfigureGraph(graphData) {
        for (const node of graphData?.nodes ?? []) {
            migrateCompactProviderNode(node);
            migratePromotedNode(node);
            if (node.type !== "H3CompactMultimodalEditDirector") continue;
            const slot = node.outputs?.findIndex(o => o.name === "credits_remaining") ?? -1;
            if (slot < 0) continue;
            const removed = new Set(node.outputs[slot].links ?? []);
            node.outputs.splice(slot, 1);
            for (const [index, output] of node.outputs.entries()) {
                if ("slot_index" in output) output.slot_index = index;
            }
            if (Array.isArray(graphData.links)) {
                graphData.links = graphData.links.filter(link => {
                    const array = Array.isArray(link);
                    const origin = array ? link[1] : link.origin_id;
                    const originSlot = array ? link[2] : link.origin_slot;
                    if (String(origin) !== String(node.id)) return true;
                    if (originSlot === slot) {
                        removed.add(array ? link[0] : link.id);
                        return false;
                    }
                    if (originSlot > slot) {
                        if (array) link[2]--; else link.origin_slot--;
                    }
                    return true;
                });
            }
            for (const target of graphData.nodes ?? []) {
                for (const input of target.inputs ?? []) {
                    if (removed.has(input.link)) input.link = null;
                }
            }
        }
    },
});
