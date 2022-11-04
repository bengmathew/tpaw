"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Validator = void 0;
const lodash_1 = __importDefault(require("lodash"));
var Validator;
(function (Validator) {
    class Failed extends Error {
        constructor(lines, path) {
            super('');
            this.lines = lodash_1.default.flatten([lines]);
            this.path = path;
        }
        get fullLines() {
            return this.path
                ? [`Property ${this.path}:`, ...this.lines.map((x) => `   ${x}`)]
                : this.lines;
        }
        get fullMessage() {
            return this.fullLines.join('\n');
        }
    }
    Validator.Failed = Failed;
    Validator.number = () => (x) => {
        if (typeof x !== 'number')
            throw new Failed('Not a number.');
        return x;
    };
    Validator.boolean = () => (x) => {
        if (typeof x !== 'boolean')
            throw new Failed('Not a boolean.');
        return x;
    };
    Validator.string = () => (x) => {
        if (typeof x !== 'string')
            throw new Failed('Not a string.');
        return x;
    };
    Validator.constant = (c) => (x) => {
        if (x !== c) {
            // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
            const message = `Not ${c === null ? 'null' : `"${c}"`}.`;
            throw new Failed(message);
        }
        return c;
    };
    Validator.array = (test) => (x) => {
        if (!lodash_1.default.isArray(x))
            throw new Failed('Not an array.');
        return x.map((e, i) => {
            try {
                return test(e);
            }
            catch (e) {
                if (e instanceof Failed) {
                    throw new Failed([
                        `At index ${i}:`,
                        ...e.fullLines.map((x) => `    ${x}`),
                    ]);
                }
                else {
                    throw e;
                }
            }
        });
    };
    Validator.object = (tests) => (x) => {
        if (!lodash_1.default.isObject(x) || lodash_1.default.isArray(x) || lodash_1.default.isFunction(x))
            throw new Failed('Not an object.');
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const anyX = x;
        const missingKeys = lodash_1.default.difference(lodash_1.default.keys(tests), lodash_1.default.keys(x));
        if (missingKeys.length > 0) {
            throw new Failed(`Missing ${missingKeys.length === 1 ? 'property' : 'properties'} ${missingKeys.join(', ')}.`);
        }
        const result = lodash_1.default.mapValues(tests, (test, key) => {
            try {
                // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-member-access
                return test(anyX[key]);
            }
            catch (e) {
                if (e instanceof Failed) {
                    throw new Failed(e.lines, `${key}${e.path ? `.${e.path}` : ''}`);
                }
                else {
                    throw e;
                }
            }
        });
        return result;
    };
    Validator.union = (...tests) => (x) => {
        const messages = [];
        let i = 0;
        for (const test of tests) {
            try {
                // eslint-disable-next-line @typescript-eslint/no-unsafe-return
                return test(x);
            }
            catch (e) {
                if (e instanceof Failed) {
                    messages.push(`Option ${i + 1}:`, ...e.fullLines.map((x) => `    ${x}`));
                }
                else {
                    throw e;
                }
            }
            i++;
        }
        throw new Failed(messages);
    };
    function intersection(...tests) {
        return (x) => {
            let result = {};
            for (const test of tests) {
                // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
                result = { ...result, ...test(x) };
            }
            // eslint-disable-next-line @typescript-eslint/no-unsafe-return
            return result;
        };
    }
    Validator.intersection = intersection;
    function chain(...tests) {
        return (x) => {
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            let result = x;
            for (const test of tests) {
                // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
                result = test(result);
            }
            // eslint-disable-next-line @typescript-eslint/no-unsafe-return
            return result;
        };
    }
    Validator.chain = chain;
})(Validator = exports.Validator || (exports.Validator = {}));
//# sourceMappingURL=Validator.js.map