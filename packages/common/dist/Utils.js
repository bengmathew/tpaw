"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.linearFnFromSlopeAndIntercept = exports.linearFnFromPointAndSlope = exports.linearFnFomPoints = exports.nundef = exports.noCase = exports.fGet = exports.assertFalse = exports.assert = void 0;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function assert(condition) {
    if (!condition) {
        throw new Error('Assert');
    }
}
exports.assert = assert;
function assertFalse() {
    throw new Error('Assert');
}
exports.assertFalse = assertFalse;
function fGet(x) {
    assert(x !== null && x !== undefined);
    return x;
}
exports.fGet = fGet;
function noCase(x) {
    // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
    throw new Error(`Didn't expect to get here: ${x}`);
}
exports.noCase = noCase;
function nundef(x) {
    assert(x !== undefined);
    return x;
}
exports.nundef = nundef;
const linearFnFomPoints = (x0, y0, x1, y1) => {
    const slope = (y1 - y0) / (x1 - x0);
    return (0, exports.linearFnFromPointAndSlope)(x0, y0, slope);
};
exports.linearFnFomPoints = linearFnFomPoints;
const linearFnFromPointAndSlope = (x, y, slope) => {
    const intercept = y - slope * x;
    return (0, exports.linearFnFromSlopeAndIntercept)(slope, intercept);
};
exports.linearFnFromPointAndSlope = linearFnFromPointAndSlope;
const linearFnFromSlopeAndIntercept = (slope, intercept) => {
    const result = (x) => slope * x + intercept;
    result.inverse = (y) => (y - intercept) / slope;
    return result;
};
exports.linearFnFromSlopeAndIntercept = linearFnFromSlopeAndIntercept;
//# sourceMappingURL=Utils.js.map