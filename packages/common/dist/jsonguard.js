"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.object = exports.chain = void 0;
const lodash_1 = __importDefault(require("lodash"));
const succeed = (value) => ({ error: false, value });
const fail = (message) => ({ error: true, message });
function chain(...guards) {
    return (x) => {
        let value = x;
        for (const guard of guards) {
            const currResult = guard(value);
            if (currResult.error)
                return currResult;
            value = currResult.value;
        }
        return succeed(value);
    };
}
exports.chain = chain;
const object = (tests) => (x) => {
    if (!lodash_1.default.isObject(x) || lodash_1.default.isArray(x) || lodash_1.default.isFunction(x))
        return fail('Not an object.');
    const missingKeys = lodash_1.default.difference(lodash_1.default.keys(tests), lodash_1.default.keys(x));
    if (missingKeys.length > 0) {
        fail(`Missing ${missingKeys.length === 1 ? 'property' : 'properties'} ${missingKeys.join(', ')}.`);
    }
    const xObj = x;
    let error = null;
    const result = lodash_1.default.mapValues(tests, (guard, key) => {
        const currResult = guard(xObj[key]);
        if (currResult.error) {
            error = `${key}: ${currResult.message}`;
        }
        return undefined;
    });
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-explicit-any
    return error ? fail(error) : succeed(result);
};
exports.object = object;
const boolean = (x) => typeof x === 'boolean' ? succeed(x) : fail('Not a boolean.');
const number = (x) => typeof x === 'number' ? succeed(x) : fail('Not a number.');
const string = (x) => typeof x === 'string' ? succeed(x) : fail('Not a string.');
const constant = (c) => (x) => 
// eslint-disable-next-line @typescript-eslint/restrict-template-expressions
x === c ? succeed(x) : fail(`Not ${c}.`);
const email = chain(string, (x) => {
    const EMAIL_REGEX = /^[^@]+@([^@]+\.[^@]+)$/;
    const DNS_REGEX = /^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$/;
    const emailMatch = EMAIL_REGEX.exec(x);
    if (emailMatch === null || !emailMatch[1])
        return fail('Email is invalid.');
    if (!DNS_REGEX.test(emailMatch[1]))
        return fail('DNS part of email is invalid');
    return succeed(x);
});
exports.default = {
    boolean,
    number,
    string,
    constant,
    email,
    chain,
};
//# sourceMappingURL=jsonguard.js.map