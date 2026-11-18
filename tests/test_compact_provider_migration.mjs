import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../web/compact_provider_migration.js", import.meta.url), "utf8");
const isolated = source.replace('import { app } from "/scripts/app.js";',
    'const app = { registerExtension() {} };');
const { migrateCompactProviderNode, migratePromotedNode } = await import("data:text/javascript," + encodeURIComponent(isolated));

for (const mode of ["Continuous Elaborate", "Elaborate Continuo", "continuous_elaborate", "Elaborate Plus"]) {
    const node = { type: "H3CompactMultimodalEditDirector",
        widgets_values: ["User request", mode, 5, "System prompt"],
        widgets_values_named: { edit_mode: mode, creative_profile: "Creative Plus" } };
    migrateCompactProviderNode(node);
    assert.deepEqual(node.widgets_values, ["User request", "Elaborate Plus", 5, "System prompt"]);
    assert.equal(node.widgets_values_named.edit_mode, "Elaborate Plus");
    assert.equal(node.widgets_values_named.creative_profile, "Creative Plus");
    const once = structuredClone(node);
    migrateCompactProviderNode(node);
    assert.deepEqual(node, once);
}
for (const mode of ["Enhance", "Elaborate", "Transfer"]) {
    const node = { type: "H3CompactMultimodalEditDirector", widgets_values: ["Request", mode] };
    migrateCompactProviderNode(node);
    assert.equal(node.widgets_values[1], mode);
}
const unrelated = { type: "OtherNode", widgets_values: ["Request", "Continuous Elaborate"] };
migrateCompactProviderNode(unrelated);
assert.equal(unrelated.widgets_values[1], "Continuous Elaborate");
console.log("Director mode migration tests passed.");

for (const [oldType, newType] of [["H3CompactMultimodalEditDirectorFix", "H3CompactMultimodalEditDirector"], ["H3CompactDirectionControlsFix", "H3CompactDirectionControls"]]) {
    const saved = {type: oldType, widgets_values: ["Normal", "Keep Guide"], inputs: [{link: 99}], outputs: [{links: [101]}]};
    const before = structuredClone(saved);
    migratePromotedNode(saved);
    assert.equal(saved.type, newType);
    assert.deepEqual(saved.widgets_values, before.widgets_values);
    assert.deepEqual(saved.inputs, before.inputs);
    assert.deepEqual(saved.outputs, before.outputs);
}
const controls = {type: "H3CompactDirectionControls", widgets_values: ["Comedy", "none", "Fast Motion", "none", "Film", "none", "Spanish", "Create New", "Keep Reference", "Auto"]};
migratePromotedNode(controls);
assert.deepEqual(controls.widgets_values, ["Comedy", "none", "Fast Motion", "Film", "Spanish", "Auto", "none", "Create New", "Auto", "Keep Reference", "Auto", "none", "Swap"]);
const once = structuredClone(controls);
migratePromotedNode(controls);
assert.deepEqual(controls, once);
console.log("Promoted node migration tests passed.");
