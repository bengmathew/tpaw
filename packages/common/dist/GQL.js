"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.API = void 0;
const json_guard_1 = require("json-guard");
var API;
(function (API) {
    const trimmed = (x) => x.trim().length === x.length ? (0, json_guard_1.succeed)(x) : (0, json_guard_1.fail)('String is not trimmed.');
    const nonEmpty = (x) => x.length > 0 ? (0, json_guard_1.succeed)(x) : (0, json_guard_1.fail)('Empty string.');
    const email = (0, json_guard_1.chain)(json_guard_1.string, trimmed, (x) => {
        const EMAIL_REGEX = /^[^@]+@([^@]+\.[^@]+)$/;
        const DNS_REGEX = /^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$/;
        const emailMatch = EMAIL_REGEX.exec(x);
        if (emailMatch === null || !emailMatch[1])
            return (0, json_guard_1.fail)('Email is invalid.');
        if (!DNS_REGEX.test(emailMatch[1]))
            return (0, json_guard_1.fail)('DNS part of email is invalid');
        return (0, json_guard_1.succeed)(x);
    });
    let SendSignInEmailInput;
    (function (SendSignInEmailInput) {
        SendSignInEmailInput.guards = { email, dest: json_guard_1.string };
        SendSignInEmailInput.check = (0, json_guard_1.object)(SendSignInEmailInput.guards);
    })(SendSignInEmailInput = API.SendSignInEmailInput || (API.SendSignInEmailInput = {}));
})(API = exports.API || (exports.API = {}));
//# sourceMappingURL=GQL.js.map