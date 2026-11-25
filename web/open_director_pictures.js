import { app } from "/scripts/app.js";

// H3 Open Director: the four fixed picture sockets (reference_image_1..4) became one growing list
// (reference_images, up to 9). A workflow saved before keeps the old sockets at the bottom of the node; this
// moves their connections, in order, into the growing list at the top and removes the old sockets.
const NODE = "H3OpenDirector";
const OLD = /^reference_image_(\d+)$/;
const GROWN = "reference_images.reference_image_";

function migrate(node) {
    if (node?.type !== NODE || !node.graph || !Array.isArray(node.inputs)) return;
    const stale = node.inputs
        .map((input, index) => ({input, index, number: Number(OLD.exec(input?.name ?? "")?.[1])}))
        .filter(item => item.number > 0)
        .sort((a, b) => a.number - b.number);
    if (!stale.length) return;
    const sources = [];
    for (const {input} of stale) {
        const link = input.link != null ? node.graph.getLink?.(input.link) ?? node.graph.links?.[input.link] : null;
        const origin = link ? node.graph.getNodeById(link.origin_id) : null;
        if (origin) sources.push([origin, link.origin_slot]);
    }
    for (const {input} of stale.slice().reverse()) {
        const index = node.inputs.indexOf(input);
        if (index >= 0) node.removeInput(index);
    }
    let next = 0;
    const connectNext = () => {
        if (next >= sources.length) {
            node.setDirtyCanvas?.(true, true);
            return;
        }
        const slot = node.inputs.findIndex(input => input.name === GROWN + next);
        if (slot < 0) return;  // the growing list was not built (old frontend): leave it to the user
        const [origin, originSlot] = sources[next];
        origin.connect(originSlot, node, slot);
        next += 1;
        requestAnimationFrame(connectNext);  // the next slot appears after this connection
    };
    requestAnimationFrame(connectNext);
}

app.registerExtension({
    name: "VisionPromptAssistant.OpenDirectorPictures",
    loadedGraphNode(node) {
        migrate(node);
    },
});
