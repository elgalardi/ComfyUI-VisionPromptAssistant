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
    const rename = value => ({deep_edit: "edit", continuous: "Enhance", "Edit Continuo": "Continuous Edit"})[value] ?? value;
    if (Array.isArray(node.widgets_values)) {
        node.widgets_values[1] = rename(node.widgets_values[1]);
    }
    if (named?.edit_mode) named.edit_mode = rename(named.edit_mode);
}

app.registerExtension({
    name: "h3.compact.externalProviderMigration",
    beforeConfigureGraph(graphData) {
        for (const node of graphData?.nodes ?? []) {
            migrateCompactProviderNode(node);
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
